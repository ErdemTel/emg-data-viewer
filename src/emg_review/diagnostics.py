"""Recording diagnostics: file integrity, timing and electrode-contact observations.

These are descriptive summaries of what is in the files. They do not validate
physiological signal quality, acquisition timing or synchronization, and nothing
here corrects a timestamp or assumes a sampling rate.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .recording import ContactTrack, EmgRecording, SegmentRun
from .schema import CHANNELS, N_CHANNELS

MIN_RATE_SPAN_S = 10.0          # shorter receipt spans: no rate estimate
MIN_RATE_GROUPS = 20
MISMATCH_TOLERANCE = 0.05       # nominal vs receipt elapsed

RATE_ESTIMATOR = (
    "received rows/s = (rows in the segment − rows in its first receipt group) / "
    "(last − first receipt time). Rows only — missing sample indices are not counted. "
    "Not estimated for spans < {:.0f} s or < {} receipt groups.").format(MIN_RATE_SPAN_S,
                                                                        MIN_RATE_GROUPS)


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


@dataclass
class RunTiming:
    run: SegmentRun
    n_rows: int
    index_first: int
    index_last: int
    missing_indices: int
    gaps: int
    duplicates: int
    decreases: int
    receipt_clock: Optional[str] = None
    receipt_span_s: Optional[float] = None
    receipt_groups: Optional[int] = None
    rows_per_group_median: Optional[float] = None
    rows_per_group_max: Optional[int] = None
    receipt_gap_ms_median: Optional[float] = None
    receipt_gap_ms_p95: Optional[float] = None
    receipt_gap_ms_max: Optional[float] = None
    received_rows_per_s: Optional[float] = None
    rate_note: str = ""
    nominal_span_s: Optional[float] = None
    nominal_index_rate: Optional[float] = None      # index steps per nominal second
    nominal_vs_receipt: Optional[float] = None      # nominal elapsed / receipt elapsed
    mismatch: bool = False
    callback_ids: Optional[int] = None              # distinct explicit callback IDs, if recorded


def run_timing(emg: EmgRecording, run: SegmentRun) -> RunTiming:
    sl = slice(run.start, run.stop)
    idx = emg.sample_index[sl]
    d = np.diff(idx)
    gaps = d > 1
    rt = RunTiming(run=run, n_rows=run.n, index_first=int(idx[0]), index_last=int(idx[-1]),
                   missing_indices=int((d[gaps] - 1).sum()), gaps=int(gaps.sum()),
                   duplicates=int((d == 0).sum()), decreases=int((d < 0).sum()))
    col = emg.receipt_column()
    if col is not None:
        t = emg.times[col][sl]
        fin = np.isfinite(t)
        tf = t[fin]
        rt.receipt_clock = col
        if len(tf) >= 2:
            rt.receipt_span_s = float(tf[-1] - tf[0])
            # receipt groups: consecutive rows with an identical receipt timestamp
            new_group = np.concatenate([[True], tf[1:] != tf[:-1]])
            starts = np.flatnonzero(new_group)
            sizes = np.diff(np.append(starts, len(tf)))
            rt.receipt_groups = int(len(starts))
            rt.rows_per_group_median = float(np.median(sizes))
            rt.rows_per_group_max = int(sizes.max())
            gaps_ms = np.diff(tf[starts]) * 1000.0
            if len(gaps_ms):
                rt.receipt_gap_ms_median = _pct(gaps_ms, 50)
                rt.receipt_gap_ms_p95 = _pct(gaps_ms, 95)
                rt.receipt_gap_ms_max = float(gaps_ms.max())
            span = float(tf[starts[-1]] - tf[starts[0]])
            if span < MIN_RATE_SPAN_S or len(starts) < MIN_RATE_GROUPS:
                rt.rate_note = (f"not estimated: receipt span {span:.2f} s / {len(starts)} receipt "
                                "groups is too short for a reliable throughput figure")
            elif not (np.diff(tf) >= 0).all():
                rt.rate_note = "not estimated: receipt times decrease within the segment"
            else:
                rt.received_rows_per_s = float((len(tf) - sizes[0]) / span)
            if not fin.all():
                rt.rate_note += f" ({int((~fin).sum())} rows without {col} ignored)"
        else:
            rt.rate_note = f"not estimated: fewer than 2 rows with {col}"
    else:
        rt.rate_note = "not estimated: no host receipt time column"
    if "device_time" in emg.times:
        dt = emg.times["device_time"][sl]
        fin = np.flatnonzero(np.isfinite(dt))
        if len(fin) >= 2:
            span = float(dt[fin[-1]] - dt[fin[0]])
            rt.nominal_span_s = span
            if span > 0:
                rt.nominal_index_rate = float((idx[fin[-1]] - idx[fin[0]]) / span)
            if rt.receipt_span_s and rt.receipt_span_s > 0 and span > 0:
                rt.nominal_vs_receipt = span / rt.receipt_span_s
                long_enough = rt.receipt_span_s >= MIN_RATE_SPAN_S
                rt.mismatch = bool(long_enough and abs(rt.nominal_vs_receipt - 1) > MISMATCH_TOLERANCE)
    if emg.callback_id is not None:
        k = emg.callback_known[sl]
        rt.callback_ids = int(len(np.unique(emg.callback_id[sl][k]))) if k.any() else 0
    return rt


def timing_warnings(timings: list) -> list:
    out = []
    for rt in timings:
        if rt.mismatch:
            out.append(
                f"{rt.run.name}: SDK nominal elapsed {rt.nominal_span_s:.2f} s vs host receipt "
                f"elapsed {rt.receipt_span_s:.2f} s (ratio {rt.nominal_vs_receipt:.3f}). "
                "Acquisition timing is unverified; timestamps are shown as recorded and no "
                "rate is assumed or corrected.")
    return out


@dataclass
class Integrity:
    rows: int
    runs: int
    extra_columns: list
    blank_lines: list
    unparsed: dict                 # column -> (count, examples)
    empty_cells: dict              # channel -> count
    nonfinite: dict                # channel -> count (incl. empty cells)
    lost_rows: Optional[int]       # None when no lost column
    lost_unknown: int
    mask_rows_partial: Optional[int]
    mask_unknown: int
    missing_indices: int
    index_gaps: int
    index_duplicates: int
    index_decreases: int
    time_nonfinite: dict = field(default_factory=dict)
    time_decreases: dict = field(default_factory=dict)
    run_warnings: list = field(default_factory=list)


def integrity(emg: EmgRecording) -> Integrity:
    t = emg.table
    timings = [run_timing(emg, r) for r in emg.runs]
    full = (1 << N_CHANNELS) - 1
    time_dec = {}
    for c, v in emg.times.items():
        dec = 0
        for r in emg.runs:
            seg = v[r.start:r.stop]
            seg = seg[np.isfinite(seg)]
            dec += int((np.diff(seg) < 0).sum())
        time_dec[c] = dec
    return Integrity(
        rows=emg.n, runs=len(emg.runs), extra_columns=list(emg.extra_columns),
        blank_lines=list(t.blank_lines),
        unparsed={c: (b.count, b.examples) for c, b in t.bad_cells.items()},
        empty_cells={c: t.empty_cells.get(c, 0) for c in CHANNELS},
        nonfinite={c: int((~np.isfinite(emg.X[:, i])).sum()) for i, c in enumerate(CHANNELS)},
        lost_rows=int(emg.lost_flag.sum()) if emg.has_lost else None,
        lost_unknown=int((~emg.lost_known).sum()),
        mask_rows_partial=(int((emg.mask_known & (emg.valid_mask != full)).sum())
                           if emg.has_mask else None),
        mask_unknown=int((~emg.mask_known).sum()),
        missing_indices=sum(x.missing_indices for x in timings),
        index_gaps=sum(x.gaps for x in timings),
        index_duplicates=sum(x.duplicates for x in timings),
        index_decreases=sum(x.decreases for x in timings),
        time_nonfinite={c: int((~np.isfinite(v)).sum()) for c, v in emg.times.items()},
        time_decreases=time_dec,
        run_warnings=list(emg.warnings),
    )


@dataclass
class ContactSummary:
    observations: int
    per_segment: dict
    receipt_clock: Optional[str]
    receipt_span_s: Optional[float]
    usable: list           # per channel count of finite, > 0 values
    unknown: list          # per channel count of empty / non-finite / non-positive values
    kohm_min: list
    kohm_median: list
    kohm_max: list
    note: str = ("Electrode–skin impedance as LOGGED by the recorder. The recorder logs "
                 "cached values when they change, so an unchanged value does not show that "
                 "a new measurement was made; measurement age is unknown.")


def contact_summary(ct: ContactTrack) -> ContactSummary:
    usable = ct.usable()
    per_seg = {}
    if ct.has_segment:
        vals, counts = np.unique(ct.segment, return_counts=True)
        per_seg = {int(v): int(c) for v, c in zip(vals, counts)}
    clock = "host_mono" if "host_mono" in ct.times else ("host_time" if "host_time" in ct.times else None)
    span = None
    if clock:
        tt = ct.times[clock][np.isfinite(ct.times[clock])]
        span = float(tt.max() - tt.min()) if len(tt) else None
    k = ct.kohm()
    mins, meds, maxs = [], [], []
    for c in range(N_CHANNELS):
        v = k[usable[:, c], c]
        mins.append(float(v.min()) if len(v) else None)
        meds.append(float(np.median(v)) if len(v) else None)
        maxs.append(float(v.max()) if len(v) else None)
    return ContactSummary(observations=ct.n, per_segment=per_seg, receipt_clock=clock,
                          receipt_span_s=span, usable=[int(x) for x in usable.sum(axis=0)],
                          unknown=[int(x) for x in (~usable).sum(axis=0)],
                          kohm_min=mins, kohm_median=meds, kohm_max=maxs)


# ----------------------------------------------------------------------------------------
# session.json
# ----------------------------------------------------------------------------------------

def session_id(meta: Optional[dict]) -> Optional[str]:
    if not meta:
        return None
    v = meta.get("session_id")
    return str(v) if v not in (None, "") else None


def is_synthetic(meta: Optional[dict]) -> bool:
    return bool(meta and meta.get("synthetic") is True)


def reported_emg_rates(meta: Optional[dict]) -> list:
    """Rates as stated in session.json, labelled with where they come from. These are
    configuration/SDK-reported values, not verified acquisition rates."""
    out = []
    if not meta:
        return out
    if isinstance(meta.get("emg_rate"), (int, float)):
        out.append(("session.json emg_rate (configured/reported)", float(meta["emg_rate"])))
    rates = meta.get("rates")
    if isinstance(rates, dict) and isinstance(rates.get("emg"), dict):
        r = rates["emg"]
        for key in ("requested_hz", "readback_hz", "device_info_hz"):
            if isinstance(r.get(key), (int, float)):
                out.append((f"session.json rates.emg.{key}", float(r[key])))
        rep = r.get("reported_hz")
        if isinstance(rep, list) and rep:
            out.append(("session.json rates.emg.reported_hz (SDK)", ", ".join(f"{x:g}" for x in rep)))
        if r.get("verified_hz") is not None:
            out.append(("session.json rates.emg.verified_hz", r.get("verified_hz")))
    return out


def session_fields(meta: Optional[dict]) -> list:
    if not meta:
        return []
    out = []
    for key in ("session_id", "name", "started", "stopped", "duration_s", "state", "schema_version",
                "app_version"):
        if key in meta:
            out.append((key, meta[key]))
    dev = meta.get("device")
    if isinstance(dev, dict):
        for key in ("name", "model", "firmware", "hardware", "backend", "package_samples"):
            if key in dev:
                out.append((f"device.{key}", dev[key]))
    if isinstance(meta.get("onboard_filters"), dict):
        on = [k for k, v in meta["onboard_filters"].items() if v]
        out.append(("onboard_filters (as configured)", ", ".join(on) if on else "none"))
    units = meta.get("units")
    if isinstance(units, dict) and "emg" in units:
        out.append(("units.emg", units["emg"]))
    if is_synthetic(meta):
        out.append(("synthetic", "YES — generated demo data, not a recording"))
    return out


def emg_unit(meta: Optional[dict]) -> Optional[str]:
    units = (meta or {}).get("units")
    if isinstance(units, dict) and isinstance(units.get("emg"), str):
        return units["emg"].split()[0]
    return None
