"""Statistics on original samples, invalid/all-invalid windows, envelope and display reduction."""

import numpy as np
import pytest

from conftest import emg_row, make_emg_text
from emg_review.stats import (channel_stats, compare_intervals, display_series, interval_wave,
                              trailing_ac_rms)


def _rows(values, lost=None, mask=None, seg=None, idx=None):
    out = []
    for i, v in enumerate(values):
        vals = list(v) if hasattr(v, "__len__") else [v] * 8
        out.append(emg_row(seg[i] if seg else 1, idx[i] if idx else i, 0.0, 0.0, 0.0,
                           lost=(lost[i] if lost else 0), mask=(mask[i] if mask else 255),
                           vals=vals))
    return out


def test_ac_rms_and_basic_stats(write, load):
    rng = np.random.default_rng(0)
    data = rng.normal(5, 2, (200, 8)).round(6)
    emg = load(write("s.csv", make_emg_text(_rows([list(r) for r in data])))).emg
    st = channel_stats(emg, 20, 120)
    for c, s in enumerate(st):
        v = data[20:120, c]
        assert s.n_rows == 100 and s.n_eligible == 100
        assert s.mean == pytest.approx(v.mean())
        assert s.ac_rms == pytest.approx(np.sqrt(np.mean((v - v.mean()) ** 2)))
        assert s.min == pytest.approx(v.min()) and s.max == pytest.approx(v.max())
        assert s.abs_peak == pytest.approx(np.abs(v).max())


def test_invalid_samples_excluded_and_counted(write, load):
    vals = [[float(i)] * 8 for i in range(10)]
    vals[3] = [None] * 8                       # lost row, empty cells
    vals[5][1] = None                          # empty ch2 with mask bit cleared
    lost = [0] * 10
    lost[3] = 1
    mask = [255] * 10
    mask[3] = 0
    mask[5] = 255 & ~0b10
    mask[7] = 255 & ~0b1                       # ch1 flagged invalid though a value exists
    emg = load(write("i.csv", make_emg_text(_rows(vals, lost, mask)))).emg
    st = channel_stats(emg, 0, 10)
    assert st[0].n_lost == 1 and st[0].n_masked == 1 and st[0].n_eligible == 8
    assert st[1].n_lost == 1 and st[1].n_masked == 1 and st[1].n_eligible == 8
    assert st[2].n_eligible == 9
    v0 = np.array([0, 1, 2, 4, 5, 6, 8, 9], float)
    assert st[0].ac_rms == pytest.approx(np.sqrt(np.mean((v0 - v0.mean()) ** 2)))
    # the masked value is still in the recording, only excluded from statistics
    assert emg.X[7, 0] == 7.0
    assert st[0].n_eligible + st[0].n_lost + st[0].n_masked + st[0].n_nonfinite == st[0].n_rows


def test_all_invalid_window_is_unknown_not_zero(write, load):
    vals = [[None] * 8 for _ in range(5)]
    emg = load(write("a.csv", make_emg_text(_rows(vals, lost=[1] * 5, mask=[0] * 5)))).emg
    for s in channel_stats(emg, 0, 5):
        assert not s.known
        assert s.ac_rms is None and s.mean is None and s.abs_peak is None


def test_unknown_validity_counted(write, load):
    text = "sample_index,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n" + "\n".join(
        f"{i}," + ",".join([str(i)] * 8) for i in range(6)) + "\n"
    emg = load(write("u.csv", text)).emg
    s = channel_stats(emg, 0, 6)[0]
    assert s.n_eligible == 6 and s.n_validity_unknown == 6


def test_envelope_resets_at_gaps_exclusions_and_segments(write, load):
    rng = np.random.default_rng(1)
    n = 60
    data = rng.normal(0, 1, (n, 8)).round(6)
    idx = list(range(30)) + list(range(35, 50)) + list(range(0, 15))   # gap after row 29
    seg = [1] * 45 + [2] * 15
    lost = [0] * n
    lost[10] = 1
    vals = [list(r) for r in data]
    vals[10] = [None] * 8
    emg = load(write("env.csv", make_emg_text(_rows(vals, lost=lost, mask=[255] * n,
                                                      seg=seg, idx=idx)))).emg
    W = 5
    e1 = trailing_ac_rms(emg, emg.runs[0], W)
    # rows 0..3 incomplete; row 10 excluded -> rows 10..14 undefined; after gap rows 30..33 undefined
    assert np.isnan(e1[:W - 1]).all() and np.isfinite(e1[W - 1:10]).all()
    assert np.isnan(e1[10:15]).all() and np.isfinite(e1[15:30]).all()
    assert np.isnan(e1[30:34]).all() and np.isfinite(e1[34:45]).all()
    # value right after the gap uses only post-gap samples
    v = data[30:35, 0]
    assert e1[34, 0] == pytest.approx(np.sqrt(np.mean((v - v.mean()) ** 2)))
    e2 = trailing_ac_rms(emg, emg.runs[1], W)
    assert np.isnan(e2[:W - 1]).all()                       # segment 2 starts fresh
    w = data[45:50, 3]
    assert e2[4, 3] == pytest.approx(np.sqrt(np.mean((w - w.mean()) ** 2)))
    assert emg.X[10, 0] != emg.X[10, 0]                     # source untouched (still NaN)


def test_display_reduction_never_bridges_breaks_and_keeps_peaks():
    n = 10_000
    x = np.arange(n, dtype=float)
    y = np.sin(x / 50)
    y[5000] = 25.0                                          # a one-sample peak
    link = np.ones(n, dtype=bool)
    link[2999] = False                                      # break between rows 2999 and 3000
    y[7000:7100] = np.nan                                   # missing interval
    xs, ys, conn = display_series(x, y, link, max_points=1000)
    assert len(xs) <= 1000 + 2 * 3 + 4
    assert ys.max() == 25.0                                 # peak preserved
    assert np.isfinite(ys).all() and np.isfinite(xs).all()
    for i in np.flatnonzero(conn[:-1]):                     # no drawn segment spans a break
        lo, hi = sorted((xs[i], xs[i + 1]))
        assert not (lo < 2999.5 < hi)
        assert not (lo < 7050 < hi)
    # raw path (few points): identical values, breaks respected
    xs2, ys2, conn2 = display_series(x[:100], y[:100], link[:100], 1000)
    assert (ys2 == y[:100]).all() and conn2[:99].all()


def test_display_reduction_with_many_breaks():
    n = 20_000
    x = np.arange(n, dtype=float)
    y = np.cos(x)
    link = np.ones(n, dtype=bool)
    link[::3] = False                                       # thousands of breaks
    xs, ys, conn = display_series(x, y, link, max_points=500)
    assert len(xs) <= 1000
    assert not conn[1::2].any()                             # break-containing bins are not linked


def test_ab_comparison_does_not_alter_or_warp(write, load):
    rng = np.random.default_rng(2)
    data = rng.normal(0, 3, (100, 8)).round(6)
    idx = list(range(50)) + list(range(53, 103))            # gap inside interval B
    emg = load(write("ab.csv", make_emg_text(_rows([list(r) for r in data], idx=idx)))).emg
    before = emg.X.copy()
    res = compare_intervals(emg, (5, 25), (40, 70))
    A, B = res["A"]["wave"], res["B"]["wave"]
    assert (A.values == data[5:25]).all() and (B.values == data[40:70]).all()
    assert list(A.offset) == list(range(20))
    assert list(B.offset[:10]) == list(range(10)) and B.offset[10] == 13   # gap kept, not stretched
    assert len(B.offset) == 30                              # no interpolation / resampling
    assert (emg.X == before).all()
    assert not A.values.flags.writeable
    assert res["A"]["stats"][0].ac_rms == channel_stats(emg, 5, 25)[0].ac_rms
    with pytest.raises(ValueError):
        emg.X[0, 0] = 1.0
    w = interval_wave(emg, 40, 70)
    assert (w.values == B.values).all()
