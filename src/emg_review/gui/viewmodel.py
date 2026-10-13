"""Shared viewer state (cursor, selection, display options) with change signals.

All row references are ORIGINAL data rows. The view model never touches recorded values.
"""

from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

from ..annotations import SelectionError, validate_range
from ..recording import AXIS_MODES, SegmentRun
from ..schema import N_CHANNELS
from ..workspace import Review

DEFAULT_ENVELOPE_WINDOW = 128
DEFAULT_CONTACT_POLICY_KOHM = (500.0, 999.0)


class ViewModel(QObject):
    reviewChanged = pyqtSignal()
    runChanged = pyqtSignal()
    axisChanged = pyqtSignal()
    cursorChanged = pyqtSignal()
    selectionChanged = pyqtSignal()
    annotationsChanged = pyqtSignal()
    displayChanged = pyqtSignal()
    abChanged = pyqtSignal()
    companionsChanged = pyqtSignal()
    dirtyChanged = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.review: Optional[Review] = None
        self._reset()

    def _reset(self):
        self.run_index = 0
        self.axis = None
        self.cursor_row: Optional[int] = None
        self.selection: Optional[tuple] = None
        self.pending_start: Optional[int] = None
        self.ab = {"A": None, "B": None}
        self.channels_visible = [True] * N_CHANNELS
        self.y_mode = "equal"
        self.envelope_on = False
        self.envelope_window = DEFAULT_ENVELOPE_WINDOW
        self.select_mode = False
        self.show_companions = True
        self.contact_policy_kohm = DEFAULT_CONTACT_POLICY_KOHM
        self.visible_rows: Optional[tuple] = None
        self.restore_rows: Optional[tuple] = None

    # -- review ------------------------------------------------------------------------
    @property
    def emg(self):
        return self.review.emg if self.review else None

    @property
    def run(self) -> Optional[SegmentRun]:
        if not self.emg:
            return None
        return self.emg.runs[self.run_index]

    def set_review(self, review: Optional[Review]):
        self.review = review
        self._reset()
        if review is not None:
            self.axis = review.emg.default_axis()
            self.apply_view_state(review.view_state or {})
        self.reviewChanged.emit()
        self.dirtyChanged.emit()

    def mark_dirty(self):
        if self.review:
            self.review._dirty = True
            self.dirtyChanged.emit()

    # -- navigation --------------------------------------------------------------------
    def set_run(self, index: int):
        if not self.emg or index == self.run_index or not 0 <= index < len(self.emg.runs):
            return
        self.run_index = index
        run = self.run
        if self.cursor_row is None or not run.contains(self.cursor_row):
            self.cursor_row = run.start
        if self.selection and not run.contains(self.selection[0]):
            self.selection = None
        self.pending_start = None
        self.visible_rows = None
        self.runChanged.emit()
        self.cursorChanged.emit()
        self.selectionChanged.emit()

    def set_axis(self, mode: str):
        if not self.emg or mode == self.axis or mode not in self.emg.available_axes():
            return
        self.axis = mode
        self.axisChanged.emit()

    def set_cursor(self, row: Optional[int]):
        if not self.emg or row is None:
            return
        row = int(min(max(row, 0), self.emg.n - 1))
        run_idx = self.emg.runs.index(self.emg.run_of_row(row))
        if run_idx != self.run_index:
            self.set_run(run_idx)
        if row != self.cursor_row:
            self.cursor_row = row
            self.cursorChanged.emit()

    def step(self, delta: int):
        if self.emg is None:
            return
        row = self.cursor_row if self.cursor_row is not None else self.run.start
        self.set_cursor(self.emg.step(row, delta))

    # -- selection ---------------------------------------------------------------------
    def set_selection(self, start: Optional[int], end_exclusive: Optional[int] = None):
        if start is None:
            if self.selection is not None or self.pending_start is not None:
                self.selection = None
                self.pending_start = None
                self.selectionChanged.emit()
            return
        validate_range(self.emg, start, end_exclusive)
        if self.selection != (start, end_exclusive):
            self.selection = (int(start), int(end_exclusive))
            self.selectionChanged.emit()

    def set_start_at_cursor(self):
        if self.cursor_row is None:
            return
        r = self.cursor_row
        if self.selection and self.emg.run_of_row(self.selection[0]).contains(r) and r < self.selection[1]:
            self.set_selection(r, self.selection[1])
        else:
            self.pending_start = r
            self.selection = (r, r + 1)
            self.selectionChanged.emit()

    def set_end_at_cursor(self):
        """The cursor row becomes the INCLUSIVE last selected row."""
        if self.cursor_row is None:
            return
        start = self.pending_start if self.pending_start is not None else (
            self.selection[0] if self.selection else None)
        if start is None:
            raise SelectionError("set a start first ( [ )")
        end = self.cursor_row + 1
        if end <= start:
            raise SelectionError("the end must not be before the start")
        self.set_selection(start, end)
        self.pending_start = None

    def set_ab(self, which: str, rows: Optional[tuple]):
        self.ab[which] = rows
        self.abChanged.emit()

    # -- display -----------------------------------------------------------------------
    def set_channel_visible(self, ch: int, on: bool):
        self.channels_visible[ch] = bool(on)
        self.displayChanged.emit()

    def set_y_mode(self, mode: str):
        self.y_mode = mode
        self.displayChanged.emit()

    def set_envelope(self, on: bool, window: int):
        self.envelope_on = bool(on)
        self.envelope_window = int(window)
        self.displayChanged.emit()

    # -- persistence -------------------------------------------------------------------
    def to_view_state(self) -> dict:
        return {
            "axis_mode": self.axis,
            "run_index": self.run_index,
            "visible_rows": list(self.visible_rows) if self.visible_rows else None,
            "cursor_row": self.cursor_row,
            "selection": list(self.selection) if self.selection else None,
            "ab": {k: (list(v) if v else None) for k, v in self.ab.items()},
            "channels_visible": list(self.channels_visible),
            "y_scale": self.y_mode,
            "envelope": {"enabled": self.envelope_on, "window_samples": self.envelope_window},
            "show_companion_tracks": self.show_companions,
            "contact_display_policy_kohm": list(self.contact_policy_kohm),
            "select_mode": self.select_mode,
        }

    def apply_view_state(self, vs: dict):
        emg = self.emg

        def rows_ok(v):
            try:
                if v and len(v) == 2:
                    validate_range(emg, int(v[0]), int(v[1]))
                    return int(v[0]), int(v[1])
            except (SelectionError, TypeError, ValueError):
                pass
            return None

        if vs.get("axis_mode") in AXIS_MODES and vs["axis_mode"] in emg.available_axes():
            self.axis = vs["axis_mode"]
        ri = vs.get("run_index")
        if isinstance(ri, int) and 0 <= ri < len(emg.runs):
            self.run_index = ri
        cr = vs.get("cursor_row")
        if isinstance(cr, int) and self.run.contains(cr):
            self.cursor_row = cr
        sel = rows_ok(vs.get("selection"))
        if sel and self.run.contains(sel[0]):
            self.selection = sel
        for k in ("A", "B"):
            self.ab[k] = rows_ok((vs.get("ab") or {}).get(k))
        cv = vs.get("channels_visible")
        if isinstance(cv, list) and len(cv) == N_CHANNELS:
            self.channels_visible = [bool(x) for x in cv]
        if vs.get("y_scale") in ("equal", "independent"):
            self.y_mode = vs["y_scale"]
        env = vs.get("envelope") or {}
        self.envelope_on = bool(env.get("enabled", False))
        w = env.get("window_samples")
        if isinstance(w, int) and w >= 2:
            self.envelope_window = w
        self.show_companions = bool(vs.get("show_companion_tracks", True))
        pol = vs.get("contact_display_policy_kohm")
        if isinstance(pol, list) and len(pol) == 2:
            self.contact_policy_kohm = (float(pol[0]), float(pol[1]))
        self.select_mode = bool(vs.get("select_mode", False))
        vr = vs.get("visible_rows")
        if isinstance(vr, list) and len(vr) == 2 and all(isinstance(x, int) for x in vr):
            self.restore_rows = (vr[0], vr[1])
