"""The synthetic demo is deterministic and exercises the required features."""

import time

import numpy as np

from conftest import sha
from emg_review.annotations import make_annotation
from emg_review.demo import demo_annotation_rows, write_demo
from emg_review.diagnostics import integrity, is_synthetic, run_timing, timing_warnings
from emg_review.stats import channel_stats, trailing_ac_rms
from emg_review.workspace import import_issues, import_paths, Review


def test_demo_is_deterministic(tmp_path):
    a = write_demo(tmp_path / "a", n_rows=5000)
    b = write_demo(tmp_path / "b", n_rows=5000)
    assert {k: sha(p) for k, p in a.items()} == {k: sha(p) for k, p in b.items()}


def test_demo_70032_rows_imports_with_expected_features(tmp_path, workarea):
    paths = write_demo(tmp_path / "demo", n_rows=70_032)
    t0 = time.perf_counter()
    out = import_paths([tmp_path / "demo"], workarea)
    t_import = time.perf_counter() - t0
    assert not out.errors and not out.rejected
    assert import_issues(out) == []
    review = Review.from_import(out)
    emg = review.emg
    assert emg.n == 70_032
    assert is_synthetic(review.session_meta)
    assert [r.segment for r in emg.runs] == [1, 2]
    integ = integrity(emg)
    assert integ.missing_indices == 40 and integ.index_gaps == 1
    assert integ.lost_rows == 2 and integ.mask_rows_partial == 3
    timings = [run_timing(emg, r) for r in emg.runs]
    assert timing_warnings(timings)                       # nominal vs receipt mismatch present
    assert 480 < timings[0].received_rows_per_s < 520
    # 16-row batches ~0.2 ms apart inside ~150 ms bursts
    assert timings[0].receipt_gap_ms_median < 5 and 140 < timings[0].receipt_gap_ms_p95 < 160
    # electrode-skin impedance: a ch3 disturbance above 999 kOhm and one unknown value
    k = review.contact.kohm()
    assert np.nanmax(k[:, 2]) > 999 and np.nanmedian(k[:, 2]) < 500
    assert (~review.contact.usable()).sum() == 1
    assert all(rows for _, rows in review.markers.emg_rows(emg))
    for s, e, kind, label in demo_annotation_rows(70_032):
        make_annotation(emg, s, e, annotation_type=kind, label=label)
    # activity interval stands out from a quiet interval in AC RMS (sanity of the generator)
    s0, e0, _, _ = demo_annotation_rows(70_032)[0]
    act = channel_stats(emg, s0, e0)[0].ac_rms
    quiet = channel_stats(emg, 100, 100 + (e0 - s0) // 4)[0].ac_rms
    assert act > 3 * quiet
    t0 = time.perf_counter()
    env = trailing_ac_rms(emg, emg.runs[0], 128)
    t_env = time.perf_counter() - t0
    assert np.isfinite(env).sum() > 0.9 * env.size
    print(f"\n[demo 70,032 rows] import+parse {t_import:.3f} s, envelope run 1 {t_env:.3f} s")
