"""Deterministic SYNTHETIC demo recording — no hardware, no private files.

Writes files in the recorder's schema (v1) so the normal import path is exercised:
emg.csv, contact.csv, imu.csv, markers.csv, session.json ("synthetic": true).

Features: 8 channels with several activity intervals; ~150 ms receipt bursts of 16-row
batches (~502 received rows/s) against a 1000 Hz SDK-nominal device_time (so nominal and
receipt elapsed disagree, as in our real recordings); a sample-index gap; lost rows and a
channel-invalid row; two segments; an electrode–skin impedance disturbance on ch3 and an
unknown impedance value; IMU; recorder markers.

Usage:  python -m emg_review.demo OUTDIR [--rows N]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .schema import CHANNELS, IMPEDANCES, IMU_AXES

DEMO_SESSION_ID = "synthetic-demo-0001"
DEMO_NAME = "SYNTHETIC DEMO (not a recording)"
NOMINAL_HZ = 1000.0          # SDK nominal rate used for device_time (as in our recorder)
RECEIVED_RATE = 502.0        # rows/s delivered (synthetic)
BATCH = 16
BURST_S = 0.150
WALL0 = 1_791_900_000.0      # fixed synthetic wall-clock anchor (deterministic)
MONO0 = 240_000.0
# (start fraction, end fraction, per-channel amplitude) of segment 1
ACTIVITY = [
    (0.08, 0.14, [40, 25, 5, 5, 8, 4, 3, 3]),
    (0.22, 0.30, [5, 6, 45, 50, 10, 6, 4, 4]),
    (0.42, 0.47, [10, 10, 10, 10, 60, 55, 10, 10]),
    (0.55, 0.66, [30, 30, 30, 30, 30, 30, 30, 30]),
    (0.78, 0.86, [4, 4, 4, 6, 8, 20, 48, 52]),
]
SEG2_ACTIVITY = [(0.30, 0.60, [20, 15, 10, 8, 8, 10, 15, 20])]
GAP_AT = 0.35                # fraction of segment 1 where 40 sample indices are missing
GAP_LEN = 40


def _fmt(v, f):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f % v


def _envelope(n, spans):
    env = np.zeros((n, 8))
    for a, b, amp in spans:
        i0, i1 = int(a * n), int(b * n)
        m = i1 - i0
        ramp = np.sin(np.linspace(0, np.pi, m)) ** 0.5
        env[i0:i1] += ramp[:, None] * np.asarray(amp, float)[None, :]
    return env


def _segment(rng, seg, n, spans, t_mono0, dev0, gap_at=None):
    """Rows of one segment: dict of arrays (index, receipt, values, flags)."""
    idx = np.arange(n, dtype=np.int64)
    if gap_at is not None:
        g = int(gap_at * n)
        idx[g:] += GAP_LEN
    # receipt: 16-row batches; batches delivered in ~150 ms bursts
    batch = np.arange(n) // BATCH
    burst = np.floor(batch * BATCH / (RECEIVED_RATE * BURST_S)).astype(int)
    n_burst = burst.max() + 1
    burst_t = np.arange(n_burst) * BURST_S + rng.normal(0, 0.003, n_burst)
    pos_in_burst = batch - batch[np.searchsorted(burst, burst)]   # batch order within its burst
    mono = t_mono0 + burst_t[burst] + pos_in_burst * 0.0002
    # keep receipt non-decreasing even with jitter
    mono = np.maximum.accumulate(mono)
    mono = np.round(mono, 6)
    dev = np.round(dev0 + idx / NOMINAL_HZ, 6)
    noise = rng.normal(0, 3.0, (n, 8))
    act = rng.normal(0, 1.0, (n, 8)) * _envelope(n, spans)
    hum = 1.5 * np.sin(2 * np.pi * 0.37 * np.arange(n)[:, None] / 50 + np.arange(8)[None, :])
    X = np.round((noise + act + hum).astype(np.float32).astype(np.float64), 6)
    lost = np.zeros(n, dtype=np.int64)
    mask = np.full(n, 255, dtype=np.int64)
    return {"segment": np.full(n, seg), "idx": idx, "mono": mono, "dev": dev, "X": X,
            "lost": lost, "mask": mask}


def write_demo(folder, n_rows: int = 70_032, seed: int = 7) -> dict:
    """Write the synthetic demo files into `folder` (created; must not contain them yet).
    Returns {kind: path}. Deterministic for a given (n_rows, seed)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n1 = int(n_rows * 0.85)
    n2 = n_rows - n1
    s1 = _segment(rng, 1, n1, ACTIVITY, MONO0, WALL0 - 120.0, gap_at=GAP_AT)
    t2 = float(s1["mono"][-1]) + 2.0           # reconnect after ~2 s
    s2 = _segment(rng, 2, n2, SEG2_ACTIVITY, t2, WALL0 + 400.0)
    # lost rows (empty channels) and a channel-invalid row in segment 1
    for r in (int(0.60 * n1), int(0.60 * n1) + 1):
        s1["lost"][r] = 1
        s1["mask"][r] = 0
        s1["X"][r, :] = np.nan
    r = int(0.70 * n1)
    s1["mask"][r] = 254
    s1["X"][r, 0] = np.nan

    segs = [s1, s2]
    emg_lines = ["segment,sample_index,device_time,host_time,host_mono,lost,valid_mask,"
                 + ",".join(CHANNELS)]
    for s in segs:
        wall = s["mono"] - MONO0 + WALL0
        for i in range(len(s["idx"])):
            vals = ",".join(_fmt(v, "%.9g") for v in s["X"][i])
            emg_lines.append(f"{s['segment'][i]},{s['idx'][i]},{s['dev'][i]:.6f},{wall[i]:.6f},"
                             f"{s['mono'][i]:.6f},{s['lost'][i]},{s['mask'][i]},{vals}")
    paths = {}
    paths["emg"] = folder / "emg.csv"
    paths["emg"].write_text("\n".join(emg_lines) + "\n")

    # electrode–skin impedance (Ohm), logged on change at ~1 s receipt cadence
    contact = ["host_time,segment," + ",".join(IMPEDANCES)]
    base = np.array([180e3, 240e3, 210e3, 150e3, 300e3, 260e3, 190e3, 220e3])
    for s in segs:
        mono = s["mono"]
        t0 = mono[0]
        last = None
        k = 0
        while True:
            t_obs = t0 + k * 1.0
            j = int(np.searchsorted(mono, t_obs))
            if j >= len(mono):
                break
            tt = mono[j] - t0
            z = base * (1 + 0.05 * np.sin(tt / 20 + np.arange(8)))
            if s is s1 and 0.45 * (mono[-1] - t0) < tt < 0.45 * (mono[-1] - t0) + 6:
                z[2] = 1.6e6                       # ch3 contact disturbance
            if s is s1 and 0.20 * (mono[-1] - t0) < tt < 0.20 * (mono[-1] - t0) + 5:
                z = last if last is not None else z  # unchanged -> recorder logs nothing
            z = np.round(z, 0)
            if last is None or not np.array_equal(z, last, equal_nan=True) or k == 0:
                cells = [_fmt(v, "%.9g") for v in z]
                if s is s1 and k == 30:
                    cells[4] = ""                  # unknown impedance on ch5
                wall = mono[j] - MONO0 + WALL0
                contact.append(f"{wall:.6f},{s['segment'][0]}," + ",".join(cells))
                last = z
            k += 1
    paths["contact"] = folder / "contact.csv"
    paths["contact"].write_text("\n".join(contact) + "\n")

    # IMU (separate stream: own indices, 50 ms receipt batches of 5 rows, nominal 200 Hz)
    imu = ["segment,sample_index,device_time,host_time,host_mono,valid_mask," + ",".join(IMU_AXES)]
    for s in segs:
        t0, t1 = s["mono"][0], s["mono"][-1]
        n_imu = int((t1 - t0) * 100)
        k = np.arange(n_imu)
        mono = np.round(t0 + 0.007 + (k // 5) * 0.050, 6)
        dev = np.round(s["dev"][0] - 0.4 + k / 200.0, 6)
        ph = k / 100.0
        acc = np.stack([0.03 + 0.05 * np.sin(ph / 3), 0.18 + 0.04 * np.sin(ph / 5),
                        -1.0 + 0.02 * np.cos(ph / 4)], axis=1)
        gyr = np.stack([2 * np.sin(ph / 2), 1.5 * np.cos(ph / 3), np.sin(ph)], axis=1)
        A = np.round(np.concatenate([acc, gyr], axis=1) + rng.normal(0, 0.01, (n_imu, 6)), 6)
        for i in range(n_imu):
            wall = mono[i] - MONO0 + WALL0
            imu.append(f"{s['segment'][0]},{1000 + k[i]},{dev[i]:.6f},{wall:.6f},{mono[i]:.6f},63,"
                       + ",".join("%.9g" % v for v in A[i]))
    paths["imu"] = folder / "imu.csv"
    paths["imu"].write_text("\n".join(imu) + "\n")

    # recorder markers: emg_sample_index = last RECEIVED EMG index at logging time
    markers = ["host_time,host_mono,segment,emg_sample_index,label"]
    for frac, label in ((0.075, "demo marker: task 1 cue"), (0.415, "demo marker: task 3 cue"),
                        (0.77, "demo marker: task 5 cue")):
        j = int(frac * n1)
        t = s1["mono"][j] + 0.010
        last_rx = int(np.searchsorted(s1["mono"], t, side="right")) - 1
        markers.append(f"{t - MONO0 + WALL0:.6f},{t:.6f},1,{s1['idx'][last_rx]},\"{label}\"")
    paths["markers"] = folder / "markers.csv"
    paths["markers"].write_text("\n".join(markers) + "\n")

    session = {
        "synthetic": True,
        "note": "SYNTHETIC DEMO generated by emg_review.demo — not a recording, no device",
        "session_id": DEMO_SESSION_ID,
        "name": DEMO_NAME,
        "app_version": "emg_review.demo",
        "emg_rate": NOMINAL_HZ,
        "imu_rate": 200.0,
        "device": {"name": "SYNTHETIC (no device)", "emg_channels": 8},
        "generator": {"seed": seed, "rows": n_rows, "received_rows_per_s": RECEIVED_RATE,
                      "batch_rows": BATCH, "burst_s": BURST_S},
        "counts": {"emg": {"written": n_rows}},
        "state": "saved",
    }
    paths["session"] = folder / "session.json"
    paths["session"].write_text(json.dumps(session, indent=2))
    return paths


def demo_annotation_rows(n_rows: int = 70_032) -> list:
    """Example user annotations for the demo: (start_row, end_row_exclusive, type, label)."""
    n1 = int(n_rows * 0.85)
    a0, a1 = int(ACTIVITY[0][0] * n1), int(ACTIVITY[0][1] * n1)
    b0, b1 = int(ACTIVITY[2][0] * n1), int(ACTIVITY[2][1] * n1)
    g = int(GAP_AT * n1)
    return [(a0, a1, "interval", "demo: ch1/ch2 activity (synthetic)"),
            (b0, b1, "interval", "demo: ch5/ch6 activity (synthetic)"),
            (g, g + 1, "bookmark", "demo: first row after a 40-index gap")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the SYNTHETIC demo recording files.")
    ap.add_argument("outdir")
    ap.add_argument("--rows", type=int, default=70_032)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    out = Path(a.outdir)
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} is not empty")
    paths = write_demo(out, a.rows, a.seed)
    for k, p in paths.items():
        print(f"{k:8s} {p}")


if __name__ == "__main__":
    main()
