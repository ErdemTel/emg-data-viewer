"""Timing semantics: nominal vs receipt mismatch is a warning, never a correction or verified rate."""

import numpy as np

from conftest import emg_row, make_emg_text
from emg_review.diagnostics import (integrity, reported_emg_rates, run_timing, timing_warnings,
                                    RATE_ESTIMATOR)
from emg_review.recording import AXIS_NOMINAL, AXIS_RECEIPT


def _bursty(n=12_000, rx_rate=502.0, nominal=1000.0, batch=16, burst=0.150, seg=1):
    rows = []
    for i in range(n):
        b = i // batch
        k = int(np.floor(b * batch / (rx_rate * burst)))
        mono = 500.0 + k * burst + (b - int(np.ceil(k * rx_rate * burst / batch))) * 0.0002
        rows.append(emg_row(seg, i, 1_000.0 + i / nominal, 1_700_000_000.0 + mono, mono))
    return rows


def test_mismatch_is_a_warning_and_timestamps_stay_as_recorded(write, load):
    rows = _bursty()
    emg = load(write("t.csv", make_emg_text(rows))).emg
    run = emg.runs[0]
    rt = run_timing(emg, run)
    # receipt throughput is ~502 rows/s; nominal index rate is 1000/s; ratio ~0.5
    assert 490 < rt.received_rows_per_s < 515
    assert rt.nominal_index_rate == np.float64(rt.nominal_index_rate)
    assert abs(rt.nominal_index_rate - 1000.0) < 1e-3
    assert rt.mismatch and 0.45 < rt.nominal_vs_receipt < 0.55
    warns = timing_warnings([rt])
    assert warns and "unverified" in warns[0] and "no rate is assumed or corrected" in warns[0]
    # nominal axis is device_time minus its first value — not rescaled, not doubled
    dev = emg.times["device_time"]
    assert np.array_equal(emg.axis_values(run, AXIS_NOMINAL), dev - dev[0])
    rec = emg.times["host_mono"]
    assert np.array_equal(emg.axis_values(run, AXIS_RECEIPT), rec - rec[0])
    # original text untouched
    assert emg.raw_row(1234)["device_time"] == rows[1234].split(",")[2]
    assert "NOMINAL" in emg.axis_title(AXIS_NOMINAL) and "unverified" in emg.axis_title(AXIS_NOMINAL)


def test_receipt_groups_are_not_called_callbacks(write, load):
    emg = load(write("g.csv", make_emg_text(_bursty(4000)))).emg
    rt = run_timing(emg, emg.runs[0])
    assert rt.receipt_groups and rt.rows_per_group_max >= 16
    assert rt.callback_ids is None          # no callback_id column -> no callback claim
    assert 140 < rt.receipt_gap_ms_p95 < 160


def test_short_spans_get_no_rate_estimate(write, load):
    emg = load(write("s.csv", make_emg_text(_bursty(1000)))).emg   # ~2 s
    rt = run_timing(emg, emg.runs[0])
    assert rt.received_rows_per_s is None and "too short" in rt.rate_note
    assert not rt.mismatch                  # no mismatch claim from a short window
    assert "missing sample indices are not counted" in RATE_ESTIMATOR


def test_missing_indices_are_not_counted_as_received(write, load):
    rows = []
    for i in range(12_000):
        idx = i if i < 6000 else i + 3000                 # 3000 missing indices
        mono = 10.0 + (i // 16) * (16 / 500.0)
        rows.append(emg_row(1, idx, idx / 1000.0, mono, mono))
    emg = load(write("m.csv", make_emg_text(rows))).emg
    rt = run_timing(emg, emg.runs[0])
    assert rt.missing_indices == 3000 and rt.gaps == 1
    assert abs(rt.received_rows_per_s - 500.0) < 1.0      # rows only
    integ = integrity(emg)
    assert integ.missing_indices == 3000 and integ.rows == 12_000


def test_reported_rates_are_labelled_not_verified():
    meta = {"emg_rate": 1000.0,
            "rates": {"emg": {"requested_hz": 1000, "reported_hz": [1000], "verified_hz": None}}}
    labels = [k for k, _ in reported_emg_rates(meta)]
    assert any("configured/reported" in k for k in labels)
    assert not any("verified" in k for k in labels)       # null verified rate is not reported


def test_callback_ids_reported_only_when_recorded(write, load):
    header = ("segment,sample_index,device_time,host_time,host_mono,lost,valid_mask,"
              "ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,callback_id")
    rows = [emg_row(1, i, i / 1000, 5 + i // 16, 5 + i // 16) + f",{i // 16}" for i in range(64)]
    emg = load(write("c.csv", header + "\n" + "\n".join(rows) + "\n")).emg
    assert run_timing(emg, emg.runs[0]).callback_ids == 4
