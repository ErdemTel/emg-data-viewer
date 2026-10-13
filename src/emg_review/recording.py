"""Parsed, read-only recording objects.

Nothing here changes a recorded value. Arrays are flagged read-only; derived
quantities (axis coordinates, eligibility, associations) are computed alongside.

Row references are zero-based DATA-ROW numbers in file order (header excluded).
Sample indices are int64. Timestamps are kept as float64 for computation, and their
exact original text is always available via EmgRecording.raw_row().
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .csvtable import CsvFormatError, CsvTable, parse_csv, read_header
from .schema import (CHANNELS, CONTACT_TYPES, EMG_TYPES, IMPEDANCES, IMU_AXES, IMU_TYPES,
                     MARKER_TYPES, N_CHANNELS, TIME_COLUMNS)

AXIS_ROW = "row"
AXIS_INDEX = "sample_index"
AXIS_RECEIPT = "receipt"
AXIS_NOMINAL = "nominal"
AXIS_MODES = (AXIS_INDEX, AXIS_ROW, AXIS_RECEIPT, AXIS_NOMINAL)

DEVICE_TIME_NOTE = ("device_time in the current recorder is the SDK NOMINAL timeline "
                    "(stream-start anchor + sample_index / nominal rate). It is not an "
                    "independently verified acquisition clock.")
RECEIPT_NOTE = ("host receipt time = when the host received the batch carrying the row "
                "(rows of one batch share it). Receipt is not acquisition time.")

KNOWN_EMG_COLUMNS = {"segment", "sample_index", "lost", "valid_mask", *TIME_COLUMNS, *CHANNELS}


def _ro(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class SourceFile:
    """One imported file: identity, fingerprint and where its immutable snapshot lives."""
    source_id: str
    kind: Optional[str]
    original_name: str
    original_path: str
    snapshot_path: Path
    sha256: str
    size: int
    imported_at: str
    columns: list = field(default_factory=list)
    n_rows: Optional[int] = None
    notes: list = field(default_factory=list)

    def manifest_entry(self) -> dict:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "original_filename": self.original_name,
            "original_path_at_import": self.original_path,
            "path": f"sources/{self.source_id}/{self.original_name}",
            "sha256": self.sha256,
            "size_bytes": self.size,
            "imported_at": self.imported_at,
            "columns": list(self.columns),
            "data_rows": self.n_rows,
            "notes": list(self.notes),
        }


@dataclass
class SegmentRun:
    """A maximal run of consecutive rows with the same recorded segment value."""
    index: int
    segment: Optional[int]      # None when the file has no segment column
    start: int                  # first row
    stop: int                   # one past the last row
    part: int = 1               # >1 when the same segment value recurs later in the file

    @property
    def n(self) -> int:
        return self.stop - self.start

    @property
    def name(self) -> str:
        seg = "segment (not recorded)" if self.segment is None else f"segment {self.segment}"
        if self.part > 1:
            seg += f" (part {self.part})"
        return seg

    def label(self) -> str:
        return f"{self.name} — rows {self.start}–{self.stop - 1} ({self.n:,} rows)"

    def contains(self, row: int) -> bool:
        return self.start <= row < self.stop


def _segment_runs(segment: Optional[np.ndarray], n: int) -> tuple:
    if segment is None:
        return [SegmentRun(0, None, 0, n)], []
    change = np.flatnonzero(np.diff(segment) != 0) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [n]])
    runs, seen, warnings = [], {}, []
    for i, (a, b) in enumerate(zip(starts, stops)):
        seg = int(segment[a])
        seen[seg] = seen.get(seg, 0) + 1
        runs.append(SegmentRun(i, seg, int(a), int(b), seen[seg]))
    for seg, count in seen.items():
        if count > 1:
            warnings.append(f"segment {seg} occurs in {count} separate row runs; they are "
                            "shown as separate parts (rows are never reordered)")
    return runs, warnings


class EmgRecording:
    """An imported EMG CSV. Values are exactly the recorded ones (NaN = empty cell)."""

    def __init__(self, source: SourceFile, table: CsvTable):
        self.source = source
        self.table = table
        n = table.n_rows
        if n == 0:
            raise CsvFormatError(f"{source.original_name}: no data rows after the header")
        self.n = n
        self.has_segment = "segment" in table.ints
        self.segment = _ro(table.ints["segment"]) if self.has_segment else None
        self.sample_index = _ro(table.ints["sample_index"])
        self.X = _ro(np.column_stack([table.floats[c] for c in CHANNELS]))
        self.times = {c: _ro(table.floats[c]) for c in TIME_COLUMNS if c in table.floats}
        self.has_lost = "lost" in table.ints
        self.has_mask = "valid_mask" in table.ints
        self.lost = _ro(table.ints["lost"]) if self.has_lost else np.zeros(n, np.int64)
        self.lost_known = (_ro(table.int_known["lost"]) if self.has_lost
                           else np.zeros(n, dtype=bool))
        self.valid_mask = _ro(table.ints["valid_mask"]) if self.has_mask else np.zeros(n, np.int64)
        self.mask_known = (_ro(table.int_known["valid_mask"]) if self.has_mask
                           else np.zeros(n, dtype=bool))
        self.callback_id = table.ints.get("callback_id")
        self.callback_known = table.int_known.get("callback_id")
        self.extra_columns = [c for c in table.columns if c not in KNOWN_EMG_COLUMNS]
        self.runs, self.warnings = _segment_runs(self.segment, n)
        self._run_of_row = np.repeat(np.arange(len(self.runs)), [r.n for r in self.runs])

        # --- recorded validity (never modifies X) -------------------------------------------
        finite = np.isfinite(self.X)
        self.lost_flag = _ro(self.lost_known & (self.lost != 0))
        bits = ((self.valid_mask[:, None] >> np.arange(N_CHANNELS)) & 1).astype(bool)
        self.mask_excluded = _ro(self.mask_known[:, None] & ~bits)
        self.eligible = _ro(finite & ~self.lost_flag[:, None] & ~self.mask_excluded)
        # validity unknown: neither lost nor valid_mask recorded for the row
        self.validity_unknown = _ro(~(self.lost_known | self.mask_known))

        # --- contiguity: row i -> i+1 is contiguous if same run and index +1 ------------------
        same_run = np.diff(self._run_of_row) == 0
        step = np.diff(self.sample_index)
        contiguous = np.zeros(n, dtype=bool)
        contiguous[:-1] = same_run & (step == 1)
        self.contiguous_after = _ro(contiguous)
        self._axis_cache = {}

    # -- basic lookups ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.source.original_name

    def run_of_row(self, row: int) -> SegmentRun:
        return self.runs[int(self._run_of_row[row])]

    def raw_row(self, row: int) -> dict:
        """Exact original cell texts of a data row (re-read from the immutable snapshot)."""
        return self.table.raw_dict(row)

    def segment_value(self, row: int):
        return int(self.segment[row]) if self.has_segment else None

    def receipt_column(self) -> Optional[str]:
        for c in ("host_mono", "host_time"):
            if c in self.times and np.isfinite(self.times[c]).any():
                return c
        return None

    def receipt_group(self, row: int) -> tuple:
        """Rows [a, b) around `row` in the same run whose receipt timestamp is identical.
        These are receipt groups only — not proven SDK callbacks."""
        col = self.receipt_column()
        run = self.run_of_row(row)
        if col is None:
            return row, row + 1
        t = self.times[col]
        v = t[row]
        if not np.isfinite(v):
            return row, row + 1
        a = row
        while a > run.start and t[a - 1] == v:
            a -= 1
        b = row + 1
        while b < run.stop and t[b] == v:
            b += 1
        return a, b

    def step(self, row: int, delta: int) -> int:
        """Move by `delta` ORIGINAL rows, staying inside the row's segment run."""
        run = self.run_of_row(row)
        return int(min(max(row + int(delta), run.start), run.stop - 1))

    def find_rows(self, sample_index: int, segment: Optional[int] = None) -> list:
        hits = np.flatnonzero(self.sample_index == sample_index)
        if segment is not None and self.has_segment:
            hits = hits[self.segment[hits] == segment]
        return [int(h) for h in hits]

    # -- axis modes ---------------------------------------------------------------------
    def available_axes(self) -> list:
        modes = [AXIS_INDEX, AXIS_ROW]
        if self.receipt_column():
            modes.append(AXIS_RECEIPT)
        if "device_time" in self.times and np.isfinite(self.times["device_time"]).any():
            modes.append(AXIS_NOMINAL)
        return modes

    def default_axis(self) -> str:
        """Recorded sample index unless it is not strictly increasing within a run."""
        for run in self.runs:
            if run.n > 1 and not (np.diff(self.sample_index[run.start:run.stop]) > 0).all():
                return AXIS_ROW
        return AXIS_INDEX

    def axis_title(self, mode: str) -> str:
        if mode == AXIS_ROW:
            return "Original data row (0-based, header excluded)"
        if mode == AXIS_INDEX:
            return "Recorded sample_index"
        if mode == AXIS_RECEIPT:
            if self.receipt_column() == "host_mono":
                return "Host receipt elapsed [s] (host_mono; receipt ≠ acquisition)"
            return "Host receipt elapsed [s] (host_time WALL-CLOCK fallback; can jump)"
        if mode == AXIS_NOMINAL:
            return "SDK NOMINAL elapsed [s] (device_time) — acquisition timing unverified"
        raise ValueError(mode)

    @staticmethod
    def axis_short(mode: str) -> str:
        return {AXIS_ROW: "Original row", AXIS_INDEX: "Sample index",
                AXIS_RECEIPT: "Host receipt time", AXIS_NOMINAL: "SDK nominal time (unverified)"}[mode]

    def axis_column(self, mode: str) -> Optional[str]:
        if mode == AXIS_RECEIPT:
            return self.receipt_column()
        if mode == AXIS_NOMINAL:
            return "device_time" if "device_time" in self.times else None
        return None

    def axis_origin(self, run: SegmentRun, mode: str):
        """Absolute value subtracted for elapsed-time axes (first finite value in the run)."""
        col = self.axis_column(mode)
        if col is None:
            return None
        t = self.times[col][run.start:run.stop]
        fin = np.flatnonzero(np.isfinite(t))
        return float(t[fin[0]]) if len(fin) else None

    def axis_values(self, run: SegmentRun, mode: str) -> np.ndarray:
        key = (run.index, mode)
        if key in self._axis_cache:
            return self._axis_cache[key]
        if mode == AXIS_ROW:
            x = np.arange(run.start, run.stop, dtype=np.float64)
        elif mode == AXIS_INDEX:
            x = self.sample_index[run.start:run.stop].astype(np.float64)
        elif mode in (AXIS_RECEIPT, AXIS_NOMINAL):
            col = self.axis_column(mode)
            if col is None:
                raise ValueError(f"axis mode {mode} is not available for this recording")
            origin = self.axis_origin(run, mode)
            t = self.times[col][run.start:run.stop]
            x = t - origin if origin is not None else np.full(run.n, np.nan)
        else:
            raise ValueError(mode)
        x = _ro(np.ascontiguousarray(x, dtype=np.float64))
        mono = bool(np.isfinite(x).all() and (run.n < 2 or (np.diff(x) >= 0).all()))
        self._axis_cache[key] = x
        self._axis_cache[(run.index, mode, "mono")] = mono
        return x

    def axis_monotonic(self, run: SegmentRun, mode: str) -> bool:
        self.axis_values(run, mode)
        return self._axis_cache[(run.index, mode, "mono")]

    def x_of_row(self, row: int, mode: str) -> float:
        run = self.run_of_row(row)
        return float(self.axis_values(run, mode)[row - run.start])

    def nearest_row(self, run: SegmentRun, mode: str, x: float) -> Optional[int]:
        xs = self.axis_values(run, mode)
        if self.axis_monotonic(run, mode):
            i = int(np.searchsorted(xs, x))
            cands = [j for j in (i - 1, i) if 0 <= j < run.n]
            # nearest; on equal distance the earlier row; within identical x the first row
            best = min(cands, key=lambda j: (abs(xs[j] - x), j))
            v = xs[best]
            best = int(np.searchsorted(xs, v, side="left"))
            return run.start + best
        d = np.abs(xs - x)
        d[~np.isfinite(d)] = np.inf
        if not np.isfinite(d).any():
            return None
        return run.start + int(np.argmin(d))

    def rows_in_range(self, run: SegmentRun, mode: str, x0: float, x1: float) -> tuple:
        """Global rows [a, b) of the run whose axis value lies in [x0, x1]. For non-monotonic
        axes, the smallest row span containing all such rows."""
        xs = self.axis_values(run, mode)
        if x1 < x0:
            x0, x1 = x1, x0
        if self.axis_monotonic(run, mode):
            a = int(np.searchsorted(xs, x0, side="left"))
            b = int(np.searchsorted(xs, x1, side="right"))
            return run.start + a, run.start + b
        hit = np.flatnonzero((xs >= x0) & (xs <= x1))
        if not len(hit):
            return run.start, run.start
        return run.start + int(hit[0]), run.start + int(hit[-1]) + 1

    # -- validity text --------------------------------------------------------------------
    def validity(self, row: int, ch: int) -> str:
        if self.lost_flag[row]:
            return "invalid (lost=1)"
        if self.mask_excluded[row, ch]:
            return "invalid (valid_mask bit cleared)"
        if not np.isfinite(self.X[row, ch]):
            return "no finite value (empty/non-numeric cell)"
        if self.validity_unknown[row]:
            return "unknown (no lost/valid_mask recorded)"
        if not self.has_lost or not self.lost_known[row] or not self.has_mask or not self.mask_known[row]:
            return "not flagged invalid (validity partly unknown)"
        return "valid per recorded flags"


# ------------------------------------------------------------------------------------------
# companion tracks
# ------------------------------------------------------------------------------------------

@dataclass
class Association:
    """A receipt-based association of a companion observation with the EMG cursor row."""
    obs_row: int
    clock: str
    obs_time: float
    emg_time: float

    @property
    def age_s(self) -> float:
        """Receipt-clock time since the LOGGED observation (not measurement age)."""
        return self.emg_time - self.obs_time


def shared_receipt_clock(emg: EmgRecording, track_times: dict) -> Optional[str]:
    for c in ("host_mono", "host_time"):
        if (c in emg.times and c in track_times and np.isfinite(emg.times[c]).any()
                and np.isfinite(track_times[c]).any()):
            return c
    return None


class _Track:
    kind = ""

    def __init__(self, source: SourceFile, table: CsvTable):
        self.source = source
        self.table = table
        self.n = table.n_rows
        self.times = {c: _ro(table.floats[c]) for c in TIME_COLUMNS if c in table.floats}
        self.has_segment = "segment" in table.ints
        self.segment = _ro(table.ints["segment"]) if self.has_segment else None
        self.segment_known = table.int_known.get("segment")

    def clock_with(self, emg: EmgRecording) -> Optional[str]:
        return shared_receipt_clock(emg, self.times)

    def association_unavailable(self, emg: EmgRecording) -> Optional[str]:
        if not self.has_segment:
            return f"{self.source.original_name} has no segment column; segment-aware association unavailable"
        if not emg.has_segment:
            return "the EMG file has no segment column; segment-aware association unavailable"
        if self.clock_with(emg) is None:
            return "no shared recorded receipt clock (host_mono/host_time) with the EMG file"
        return None

    def _same_segment(self, emg: EmgRecording, run: SegmentRun) -> np.ndarray:
        """Observation rows whose recorded segment equals the run's segment value. If the
        segment value recurs in several runs, observations are further limited to the
        run's receipt-time span (closed at the start, up to the next run's start)."""
        ok = (self.segment == run.segment)
        if self.segment_known is not None:
            ok &= self.segment_known
        parts = [r for r in emg.runs if r.segment == run.segment]
        if len(parts) > 1:
            clock = self.clock_with(emg)
            t = self.times[clock]
            et = emg.times[clock]
            lo = np.nanmin(et[run.start:run.stop]) if run.part > 1 else -np.inf
            nxt = [r for r in parts if r.part == run.part + 1]
            hi = np.nanmin(et[nxt[0].start:nxt[0].stop]) if nxt else np.inf
            ok &= (t >= lo) & (t < hi)
        return np.flatnonzero(ok)

    def asof(self, emg: EmgRecording, row: int) -> tuple:
        """Backward as-of match: the last LOGGED observation in the same segment whose
        receipt time <= the EMG row's receipt time. Returns (Association | None, reason)."""
        why = self.association_unavailable(emg)
        if why:
            return None, why
        clock = self.clock_with(emg)
        t_row = emg.times[clock][row]
        if not np.isfinite(t_row):
            return None, f"EMG row {row} has no {clock} value"
        run = emg.run_of_row(row)
        cand = self._same_segment(emg, run)
        if not len(cand):
            return None, f"no logged observation in {run.name}"
        t = self.times[clock][cand]
        ok = np.isfinite(t) & (t <= t_row)
        if not ok.any():
            return None, f"no observation logged in {run.name} at or before this row's receipt"
        tc = t[ok]
        idx = cand[ok]
        best = np.flatnonzero(tc == tc.max())[-1]       # latest; ties -> last in file order
        return Association(int(idx[best]), clock, float(tc[best]), float(t_row)), ""

    def placement(self, emg: EmgRecording, run: SegmentRun) -> tuple:
        """Place observations of this run's segment on the EMG rows: each observation goes
        to the FIRST EMG row (same run) whose receipt time >= the observation's receipt time,
        i.e. the first row from which the backward as-of association can select it.
        Returns (obs_rows, emg_rows, n_unplaced)."""
        if self.association_unavailable(emg):
            return np.empty(0, np.int64), np.empty(0, np.int64), 0
        clock = self.clock_with(emg)
        et = emg.times[clock][run.start:run.stop]
        if not (np.isfinite(et).all() and (np.diff(et) >= 0).all()):
            return np.empty(0, np.int64), np.empty(0, np.int64), 0
        cand = self._same_segment(emg, run)
        t = self.times[clock][cand]
        fin = np.isfinite(t)
        cand, t = cand[fin], t[fin]
        pos = np.searchsorted(et, t, side="left")
        ok = pos < run.n
        return cand[ok], run.start + pos[ok], int((~ok).sum() + (~fin).sum())

    def receipt_x(self, emg: EmgRecording, run: SegmentRun) -> Optional[tuple]:
        """Exact receipt-axis coordinates when the track shares the EMG receipt-axis clock."""
        col = emg.receipt_column()
        if col is None or col not in self.times or self.association_unavailable(emg):
            return None
        origin = emg.axis_origin(run, AXIS_RECEIPT)
        rows = self._same_segment(emg, run)
        x = self.times[col][rows] - origin
        return rows, x


class ContactTrack(_Track):
    """contact.csv: ELECTRODE–SKIN impedance observations (not object contact).
    Values are the recorded Ohm values; kOhm is a display unit conversion only."""
    kind = "contact"

    def __init__(self, source, table):
        super().__init__(source, table)
        self.Z = _ro(np.column_stack([table.floats[c] for c in IMPEDANCES]))

    def kohm(self) -> np.ndarray:
        return self.Z / 1000.0

    def usable(self) -> np.ndarray:
        """Observation is a usable impedance value: finite and > 0. Anything else is unknown."""
        return np.isfinite(self.Z) & (self.Z > 0)


class ImuTrack(_Track):
    kind = "imu"

    def __init__(self, source, table):
        super().__init__(source, table)
        self.A = _ro(np.column_stack([table.floats[c] for c in IMU_AXES]))
        self.sample_index = table.ints.get("sample_index")


class MarkerTable(_Track):
    """Recorder markers: read-only source annotations. emg_sample_index is the LAST
    RECEIVED EMG sample index when the marker was logged — not an acquisition-time match."""
    kind = "markers"

    def __init__(self, source, table):
        super().__init__(source, table)
        self.labels = table.texts.get("label", [""] * self.n)
        self.emg_index = table.ints.get("emg_sample_index")
        self.emg_index_known = table.int_known.get("emg_sample_index")

    def emg_rows(self, emg: EmgRecording) -> list:
        """For each marker: (marker_row, list of EMG rows with that segment+sample_index)."""
        out = []
        for i in range(self.n):
            if self.emg_index is None or not self.emg_index_known[i]:
                out.append((i, []))
                continue
            seg = int(self.segment[i]) if self.has_segment and self.segment_known[i] else None
            out.append((i, emg.find_rows(int(self.emg_index[i]), seg)))
        return out


# ------------------------------------------------------------------------------------------
# parsing entry points (GUI-free)
# ------------------------------------------------------------------------------------------

def parse_emg(source: SourceFile) -> EmgRecording:
    cols = read_header(source.snapshot_path)
    required = ["sample_index", *CHANNELS] + (["segment"] if "segment" in cols else [])
    table = parse_csv(source.snapshot_path, EMG_TYPES, required=tuple(required))
    return EmgRecording(source, table)


def parse_contact(source: SourceFile) -> ContactTrack:
    cols = read_header(source.snapshot_path)
    required = list(IMPEDANCES) + (["segment"] if "segment" in cols else [])
    return ContactTrack(source, parse_csv(source.snapshot_path, CONTACT_TYPES, tuple(required)))


def parse_imu(source: SourceFile) -> ImuTrack:
    cols = read_header(source.snapshot_path)
    required = list(IMU_AXES) + (["segment"] if "segment" in cols else [])
    return ImuTrack(source, parse_csv(source.snapshot_path, IMU_TYPES, tuple(required)))


def parse_markers(source: SourceFile) -> MarkerTable:
    return MarkerTable(source, parse_csv(source.snapshot_path, MARKER_TYPES, ("label",)))
