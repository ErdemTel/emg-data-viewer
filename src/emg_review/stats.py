"""Interval statistics, derived activity envelope and display-only reduction.

All statistics read the ORIGINAL samples of the recording. Nothing here modifies,
filters, resamples or normalizes the source data; results are separate derived values.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .recording import EmgRecording, SegmentRun
from .schema import N_CHANNELS

EXCLUSION_POLICY = (
    "Eligible sample = finite recorded value, row not flagged lost (lost≠0), and the "
    "channel's bit set in valid_mask. Rows without a recorded lost/valid_mask value are "
    "not excluded (their validity is unknown and counted separately). Excluded samples "
    "stay in the source recording; they are only left out of these statistics.")
AC_RMS_DEFINITION = "AC RMS = sqrt(mean((x − mean(x))²)) over the eligible samples"
ENVELOPE_DEFINITION = (
    "Derived activity envelope (display only): trailing AC RMS over the last W eligible "
    "samples, where W is a SAMPLE COUNT (not milliseconds; acquisition timing is unverified). "
    "It restarts after every segment boundary, sample-index discontinuity or excluded "
    "sample, and is undefined (not drawn) until W contiguous eligible samples exist.")


@dataclass
class ChannelStats:
    channel: int                # 0-based
    n_rows: int                 # rows in the interval
    n_eligible: int
    n_lost: int                 # excluded: row flagged lost
    n_masked: int               # excluded: valid_mask bit cleared (and not lost)
    n_nonfinite: int            # excluded: empty / non-numeric / inf (and not lost/masked)
    n_validity_unknown: int     # eligible rows whose lost/valid_mask were not recorded
    mean: Optional[float]
    ac_rms: Optional[float]
    min: Optional[float]
    max: Optional[float]
    abs_peak: Optional[float]

    @property
    def known(self) -> bool:
        return self.n_eligible > 0


def channel_stats(emg: EmgRecording, start_row: int, end_row_exclusive: int,
                  channels=None) -> list:
    """Per-channel statistics over original rows [start_row, end_row_exclusive)."""
    if not 0 <= start_row <= end_row_exclusive <= emg.n:
        raise ValueError(f"row range [{start_row}, {end_row_exclusive}) outside 0..{emg.n}")
    sl = slice(start_row, end_row_exclusive)
    X = emg.X[sl]
    elig = emg.eligible[sl]
    lost = emg.lost_flag[sl]
    masked = emg.mask_excluded[sl] & ~lost[:, None]
    nonfinite = ~np.isfinite(X) & ~lost[:, None] & ~masked
    unknown = emg.validity_unknown[sl]
    out = []
    for c in (range(N_CHANNELS) if channels is None else channels):
        v = X[elig[:, c], c]
        n_el = int(v.size)
        if n_el:
            mean = float(v.mean())
            ac = float(np.sqrt(np.mean((v - mean) ** 2)))
            mn, mx = float(v.min()), float(v.max())
            peak = float(np.abs(v).max())
        else:
            mean = ac = mn = mx = peak = None
        out.append(ChannelStats(
            channel=c, n_rows=end_row_exclusive - start_row, n_eligible=n_el,
            n_lost=int(lost.sum()), n_masked=int(masked[:, c].sum()),
            n_nonfinite=int(nonfinite[:, c].sum()),
            n_validity_unknown=int((unknown & elig[:, c]).sum()),
            mean=mean, ac_rms=ac, min=mn, max=mx, abs_peak=peak))
    return out


def trailing_ac_rms(emg: EmgRecording, run: SegmentRun, window: int) -> np.ndarray:
    """Derived envelope for the rows of `run`: (run.n, 8) array, NaN where undefined.
    Value at row i = AC RMS of rows i-W+1..i if all W rows are eligible and consecutive
    (same run, sample_index +1 each step); otherwise NaN. Never fills or bridges gaps."""
    window = int(window)
    if window < 2:
        raise ValueError("envelope window must be at least 2 samples")
    a, b = run.start, run.stop
    X = emg.X[a:b]
    elig = emg.eligible[a:b]
    contig = emg.contiguous_after[a:b].copy()
    out = np.full((run.n, N_CHANNELS), np.nan)
    for c in range(N_CHANNELS):
        ok = elig[:, c]
        # prev_link[i]: row i continues an eligible, contiguous stretch from row i-1
        prev_link = np.zeros(run.n, dtype=bool)
        prev_link[1:] = contig[:-1] & ok[:-1] & ok[1:]
        next_link = np.append(prev_link[1:], False)
        starts = np.flatnonzero(ok & ~prev_link)
        stops = np.flatnonzero(ok & ~next_link) + 1
        for s, e in zip(starts, stops):
            # stretch [s, e) of contiguous eligible samples
            if e - s < window:
                continue
            v = X[s:e, c]
            v = v - v.mean()                     # constant shift; AC RMS is shift-invariant
            c1 = np.concatenate([[0.0], np.cumsum(v)])
            c2 = np.concatenate([[0.0], np.cumsum(v * v)])
            s1 = c1[window:] - c1[:-window]
            s2 = c2[window:] - c2[:-window]
            var = s2 / window - (s1 / window) ** 2
            out[s + window - 1:e, c] = np.sqrt(np.maximum(var, 0.0))
    return out


@dataclass
class IntervalWave:
    """Original samples of an interval for comparison plots. `offset` is the recorded
    sample_index minus the interval's first sample_index — a display coordinate only
    (gaps stay gaps; nothing is stretched, interpolated or normalized)."""
    start_row: int
    end_row_exclusive: int
    offset: np.ndarray        # int64
    values: np.ndarray        # (n, 8) read-only view of the recorded values
    eligible: np.ndarray      # (n, 8) bool


def interval_wave(emg: EmgRecording, start_row: int, end_row_exclusive: int) -> IntervalWave:
    sl = slice(start_row, end_row_exclusive)
    idx = emg.sample_index[sl]
    return IntervalWave(start_row, end_row_exclusive, idx - idx[0], emg.X[sl], emg.eligible[sl])


def compare_intervals(emg: EmgRecording, a: tuple, b: tuple) -> dict:
    """A/B comparison within one recording with identical settings for both intervals."""
    return {"A": {"rows": a, "stats": channel_stats(emg, *a), "wave": interval_wave(emg, *a)},
            "B": {"rows": b, "stats": channel_stats(emg, *b), "wave": interval_wave(emg, *b)}}


# --------------------------------------------------------------------------------------
# display-only reduction (never used for statistics or inspection)
# --------------------------------------------------------------------------------------

def display_series(x: np.ndarray, y: np.ndarray, link: np.ndarray, max_points: int) -> tuple:
    """Prepare one channel for drawing.

    x, y: coordinates for consecutive rows; y NaN = not drawn. link[i] True when a line may
    connect row i to row i+1 (same run, consecutive sample index). Returns (xs, ys, connect)
    for a pyqtgraph curve with an explicit connect array; NaN-free.

    When there are more rows than max_points, rows are grouped into bins that never span a
    break, and each bin is drawn as its min and max (peak-preserving). This is purely a
    drawing reduction — it is not a resampled recording.
    """
    n = len(y)
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int32)
    finite = np.isfinite(y) & np.isfinite(x)
    if n <= max_points:
        conn = np.zeros(n, dtype=bool)
        conn[:-1] = link[:-1] & finite[:-1] & finite[1:]
        return (np.where(finite, x, 0.0), np.where(finite, y, 0.0), conn.astype(np.int32))
    per_bin = int(np.ceil(n / max(1, max_points // 2)))
    regular = np.arange(0, n, per_bin)
    # piece boundaries: a forbidden link, or a change between drawable and missing values
    # (a run of missing rows becomes one undrawn bin, not one bin per row)
    breaks = np.flatnonzero(~link[:-1] | (finite[1:] != finite[:-1])) + 1
    if len(breaks) <= max_points:
        edges = np.union1d(regular, breaks).astype(np.int64)
        edges = edges[edges < n]
        ends = np.append(edges[1:], n)
        inner_break = np.zeros(len(edges), dtype=bool)
    else:
        # too many breaks to split on: bins that contain a break are drawn unconnected
        edges = regular
        ends = np.append(edges[1:], n)
        inner_break = (np.searchsorted(breaks, ends, side="left")
                       - np.searchsorted(breaks, edges, side="right")) > 0
    yy = np.where(finite, y, np.nan)
    with np.errstate(invalid="ignore"):
        lo = np.fmin.reduceat(yy, edges)
        hi = np.fmax.reduceat(yy, edges)
    xm = 0.5 * (x[edges] + x[ends - 1])
    ok = np.isfinite(lo) & np.isfinite(xm)
    # bin j connects to bin j+1 only if there is no break at or inside their boundary
    boundary_linked = np.zeros(len(edges), dtype=bool)
    if len(edges) > 1:
        boundary_linked[:-1] = (np.isin(edges[1:], breaks, invert=True)
                                & ~inner_break[:-1] & ~inner_break[1:])
    xs = np.repeat(xm, 2)
    ys = np.empty(2 * len(edges))
    ys[0::2] = lo
    ys[1::2] = hi
    conn = np.zeros(2 * len(edges), dtype=bool)
    conn[0::2] = ok                                         # min -> max inside a bin
    nxt_ok = np.append(ok[1:], False)
    conn[1::2] = boundary_linked & ok & nxt_ok              # max -> next bin's min
    xs = np.where(np.repeat(ok, 2), xs, 0.0)
    ys = np.where(np.repeat(ok, 2), ys, 0.0)
    return xs, ys, conn.astype(np.int32)
