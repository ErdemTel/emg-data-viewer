"""Right-hand inspector: exact sample, selection statistics, annotations, A/B comparison and
recording diagnostics. Every number here is computed from the original rows."""

import html

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout,
                             QHeaderView, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QTabWidget, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from ..annotations import TYPE_BOOKMARK
from ..diagnostics import (RATE_ESTIMATOR, contact_summary, integrity, is_synthetic,
                           reported_emg_rates, run_timing, session_fields, timing_warnings)
from ..recording import DEVICE_TIME_NOTE, RECEIPT_NOTE
from ..schema import CHANNELS, KIND_LABELS, N_CHANNELS, TIME_COLUMNS
from ..stats import (AC_RMS_DEFINITION, ENVELOPE_DEFINITION, EXCLUSION_POLICY, channel_stats,
                     compare_intervals, display_series)
from .plots import A_BRUSH, B_BRUSH, CH_COLORS

TIME_NOTES = {"device_time": "SDK NOMINAL timeline — not a verified acquisition clock",
              "host_time": "wall-clock receipt time (can jump)",
              "host_mono": "monotonic receipt time"}
GOOD, MARGINAL, POOR, UNKNOWN = (QColor(200, 240, 200), QColor(255, 225, 180),
                                 QColor(255, 200, 200), QColor(225, 225, 225))


def fmt(v, digits=4):
    if v is None:
        return "unknown"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "unknown"
        return f"{v:.{digits}g}"
    return str(v)


def _table(cols, stretch_last=True):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setWordWrap(True)
    h = t.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.ResizeToContents)
    if stretch_last:
        h.setStretchLastSection(True)
    return t


def _item(text, tip=None, color=None, bold=False):
    it = QTableWidgetItem(text)
    if tip:
        it.setToolTip(tip)
    if color is not None:
        it.setBackground(QBrush(color))
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    return it


def contact_level(kohm, policy):
    """Display policy colour (configurable; recorder convention 500/999 kΩ)."""
    if kohm is None or not np.isfinite(kohm) or kohm <= 0:
        return UNKNOWN, "unknown"
    lo, hi = policy
    if kohm <= lo:
        return GOOD, f"≤ {lo:g} kΩ"
    if kohm <= hi:
        return MARGINAL, f"≤ {hi:g} kΩ"
    return POOR, f"> {hi:g} kΩ"


# ------------------------------------------------------------------------------------------

class SamplePanel(QWidget):
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        lay = QVBoxLayout(self)
        self.head = QLabel("Click a plot or use ←/→ to inspect original rows.")
        self.head.setWordWrap(True)
        lay.addWidget(self.head)
        self.table = _table(["Field", "Value (exact original text)"])
        lay.addWidget(self.table)
        for sig in (vm.cursorChanged, vm.reviewChanged, vm.axisChanged, vm.companionsChanged,
                    vm.displayChanged):
            sig.connect(self.refresh)

    def refresh(self):
        vm = self.vm
        emg = vm.emg
        self.table.setRowCount(0)
        if emg is None or vm.cursor_row is None:
            return
        r = vm.cursor_row
        run = emg.run_of_row(r)
        raw = emg.raw_row(r)
        rows = []
        add = lambda k, v, tip=None, color=None, bold=False: rows.append((k, v, tip, color, bold))
        add("Original data row", f"{r}  (0-based, header excluded)", bold=True)
        add("Segment", f"{run.name} (rows {run.start}–{run.stop - 1})")
        add("sample_index", raw["sample_index"])
        if r > run.start:
            d = int(emg.sample_index[r] - emg.sample_index[r - 1])
            txt = ("+1" if d == 1 else f"+{d} (index gap: {d - 1} missing)" if d > 1
                   else "0 (duplicate index)" if d == 0 else f"{d} (index decreased)")
        else:
            txt = "first row of the segment"
        add("Δ sample_index vs previous row", txt)
        for col in TIME_COLUMNS:
            if col not in emg.times:
                add(col, "(column not recorded — unknown)")
                continue
            add(col, raw[col] if raw[col].strip() else "(empty — unknown)", TIME_NOTES[col])
            if r > run.start:
                dt = (emg.times[col][r] - emg.times[col][r - 1]) * 1000.0
                add(f"Δ {col} vs previous row", f"{dt:.3f} ms" if np.isfinite(dt) else "unknown",
                    TIME_NOTES[col])
        x = emg.x_of_row(r, vm.axis)
        add("Position on current axis", f"{fmt(x, 10)}  ({emg.axis_short(vm.axis)})")
        a, b = emg.receipt_group(r)
        col = emg.receipt_column()
        if col:
            add("Receipt group", f"rows {a}–{b - 1} ({b - a} rows share this {col})",
                "Identical receipt timestamps define a receipt group only; it is not shown to be "
                "one SDK callback unless explicit callback IDs are recorded.")
        if "callback_id" in raw:
            add("callback_id (recorded)", raw["callback_id"] or "(empty — unknown)")
        if emg.has_lost:
            add("lost", raw["lost"] if raw["lost"].strip() else "(empty — unknown)")
        else:
            add("lost", "(column not recorded — unknown)")
        if emg.has_mask and emg.mask_known[r]:
            v = int(emg.valid_mask[r])
            add("valid_mask", f"{raw['valid_mask']}  = {v:08b} (bit c = ch c+1)")
        else:
            add("valid_mask", "(not recorded — unknown)")
        for c, ch in enumerate(CHANNELS):
            v = emg.validity(r, c)
            color = POOR if v.startswith("invalid") or v.startswith("no finite") else (
                UNKNOWN if "unknown" in v else None)
            add(ch, f"{raw[ch] if raw[ch].strip() else '(empty)'}   — {v}", color=color)
        for col in emg.extra_columns:
            add(f"{col} (extra column)", raw[col])
        rv = vm.review
        if rv.contact is not None:
            m, why = rv.contact.asof(emg, r)
            add("ELECTRODE–SKIN IMPEDANCE", "receipt-based association (backward as-of, same "
                "segment)", bold=True)
            if m is None:
                add("as-of observation", f"none — {why}", color=UNKNOWN)
            else:
                craw = rv.contact.table.raw_dict(m.obs_row)
                add("logged observation", f"contact row {m.obs_row}, {m.clock} = {craw.get(m.clock, '')}")
                add("time since LOGGED observation", f"{m.age_s * 1000:.1f} ms on the {m.clock} receipt clock",
                    "Receipt-clock time since the observation was LOGGED. The recorder logs cached "
                    "values when they change, so this is not the age of the impedance measurement "
                    "(measurement age unknown).")
                kohm = rv.contact.kohm()[m.obs_row]
                for c in range(N_CHANNELS):
                    color, lvl = contact_level(kohm[c] if rv.contact.usable()[m.obs_row, c] else None,
                                               vm.contact_policy_kohm)
                    txt = (f"{kohm[c]:.1f} kΩ  (recorded {craw.get(f'imp{c + 1}', '')} Ω)  — display "
                           f"policy {lvl}" if lvl != "unknown"
                           else f"unknown (recorded {craw.get(f'imp{c + 1}', '')!r})")
                    add(f"imp{c + 1}", txt, "Colour = configurable display policy, not a "
                        "universal validity rule; waveform inspection is independent of it.", color)
                for extra in ("observation", "source", "measurement_age_s", "since_change_s"):
                    if extra in craw:
                        add(f"contact.{extra} (recorded)", craw[extra] or "(empty)")
        self.table.setRowCount(len(rows))
        for i, (k, v, tip, color, bold) in enumerate(rows):
            self.table.setItem(i, 0, _item(k, tip, None, bold))
            self.table.setItem(i, 1, _item(v, tip, color, bold))
        self.table.resizeRowsToContents()
        self.head.setText(f"Inspecting original row {r} of {emg.n:,} "
                          f"({'synthetic demo' if is_synthetic(rv.session_meta) else rv.emg.name}).")


# ------------------------------------------------------------------------------------------

def _bars(plot, values, x_offsets=(0.0,), width=0.8, brushes=None):
    plot.clear()
    for k, vals in enumerate(values):
        x = np.arange(N_CHANNELS) + 1 + x_offsets[k]
        ok = np.array([v is not None for v in vals])
        h = np.array([v if v is not None else 0.0 for v in vals])
        if ok.any():
            plot.addItem(pg.BarGraphItem(x=x[ok], height=h[ok], width=width,
                                         brushes=[brushes[k] if brushes else pg.mkBrush(CH_COLORS[c])
                                                  for c in np.flatnonzero(ok)]))
        for c in np.flatnonzero(~ok):
            t = pg.TextItem("unknown", color=(120, 120, 120), anchor=(0.5, 1))
            t.setPos(x[c], 0)
            plot.addItem(t)
    plot.getAxis("bottom").setTicks([[(c + 1, f"ch{c + 1}") for c in range(N_CHANNELS)]])


class SelectionPanel(QWidget):
    def __init__(self, vm, actions):
        super().__init__()
        self.vm = vm
        lay = QVBoxLayout(self)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        lay.addWidget(self.info)
        grid = QGridLayout()
        names = ["set_start", "set_end", "clear_sel", "zoom_sel", "label_sel", "bookmark",
                 "store_a", "store_b"]
        for i, n in enumerate(names):
            b = QToolButton()
            b.setDefaultAction(actions[n])
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            b.setSizePolicy(b.sizePolicy().Expanding, b.sizePolicy().Fixed)
            grid.addWidget(b, i // 4, i % 4)
        lay.addLayout(grid)
        self.table = _table(["ch", "eligible / rows", "mean", "AC RMS", "min", "max", "|peak|",
                             "excluded lost / mask / non-finite", "validity unknown"], False)
        lay.addWidget(self.table, 2)
        pol = QLabel(f"<small>{html.escape(AC_RMS_DEFINITION)}. {html.escape(EXCLUSION_POLICY)} "
                     "All-excluded channels show <b>unknown</b>, not 0. Statistics use every original "
                     "sample in the selection (never the display reduction). Amplitude alone does not "
                     "indicate gesture, force, stiffness or signal quality.</small>")
        pol.setWordWrap(True)
        lay.addWidget(pol)
        self.bar = pg.PlotWidget()
        self.bar.setMinimumHeight(150)
        self.bar.setMenuEnabled(False)
        self.bar.setMouseEnabled(False, False)
        self.bar.hideButtons()
        self.bar.setTitle("<span style='font-size:9pt'>Selected-interval AC RMS per channel (same "
                          "computation as the table)</span>")
        lay.addWidget(self.bar, 1)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.refresh)
        vm.selectionChanged.connect(self._timer.start)
        vm.reviewChanged.connect(self._timer.start)
        vm.runChanged.connect(self._timer.start)

    def refresh(self):
        vm = self.vm
        emg = vm.emg
        self.table.setRowCount(0)
        self.bar.clear()
        if emg is None or vm.selection is None:
            pend = (f" Start set at row {vm.pending_start}; move the cursor and press ] to set the "
                    "inclusive end." if vm.pending_start is not None else "")
            self.info.setText("<b>No selection.</b> Shift+drag (or Select mode, S) on the plots, or "
                              "use Set Start [ / Set End ] at the cursor." + pend)
            return
        a, b = vm.selection
        run = emg.run_of_row(a)
        si, sl = emg.raw_row(a)["sample_index"], emg.raw_row(b - 1)["sample_index"]
        pend = (" <i>(start set; press ] at the cursor to set the inclusive end)</i>"
                if vm.pending_start is not None else "")
        self.info.setText(
            f"<b>Selection:</b> original rows [{a}, {b}) — <b>{b - a:,} rows</b>; first row {a} "
            f"(sample_index {si}), <b>inclusive last row {b - 1}</b> (sample_index {sl}); "
            f"{run.name}.{pend}")
        st = channel_stats(emg, a, b)
        self.table.setRowCount(N_CHANNELS)
        for c, s in enumerate(st):
            vals = [f"ch{c + 1}", f"{s.n_eligible:,} / {s.n_rows:,}", fmt(s.mean), fmt(s.ac_rms),
                    fmt(s.min), fmt(s.max), fmt(s.abs_peak),
                    f"{s.n_lost} / {s.n_masked} / {s.n_nonfinite}", f"{s.n_validity_unknown:,}"]
            for j, v in enumerate(vals):
                self.table.setItem(c, j, _item(v, color=UNKNOWN if not s.known and j else None))
        _bars(self.bar, [[s.ac_rms for s in st]])


# ------------------------------------------------------------------------------------------

class AnnotationsPanel(QWidget):
    jumpRequested = pyqtSignal(str)        # annotation id, or "marker:<row>"
    editRequested = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(self, vm, actions):
        super().__init__()
        self.vm = vm
        lay = QVBoxLayout(self)
        note = QLabel("<small>User annotations use half-open original row ranges [start, end); "
                      "the inclusive last row is end − 1. Labels are user assessments, not "
                      "verified events. Recorder markers (grey) are read-only source annotations: "
                      "their emg_sample_index is the last RECEIVED EMG sample when logged.</small>")
        note.setWordWrap(True)
        lay.addWidget(note)
        self.table = _table(["Type", "Rows [start, end)", "Last row", "Segment", "Label",
                             "Status", "Source"])
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemDoubleClicked.connect(lambda *_: self._emit(self.jumpRequested))
        lay.addWidget(self.table)
        row = QHBoxLayout()
        for text, sig in (("Jump to", self.jumpRequested), ("Edit…", self.editRequested),
                          ("Delete", self.deleteRequested)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=sig: self._emit(s))
            row.addWidget(b)
        for n in ("bookmark", "label_sel", "export"):
            b = QToolButton()
            b.setDefaultAction(actions[n])
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            row.addWidget(b)
        lay.addLayout(row)
        vm.annotationsChanged.connect(self.refresh)
        vm.reviewChanged.connect(self.refresh)
        vm.companionsChanged.connect(self.refresh)

    def _emit(self, sig):
        r = self.table.currentRow()
        if r < 0:
            return
        sig.emit(self.table.item(r, 0).data(Qt.UserRole))

    def refresh(self):
        vm = self.vm
        rv = vm.review
        self.table.setRowCount(0)
        if rv is None:
            return
        entries = []
        for a in rv.annotations.sorted():
            seg = a.segment or "—"
            entries.append((a.annotation_id, a.annotation_type, f"[{a.start_row}, {a.end_row_exclusive})",
                            str(a.last_row), seg, a.label, a.review_status, a.annotation_source, None))
        if rv.markers is not None:
            for mrow, rows in rv.markers.emg_rows(rv.emg):
                where = f"row {rows[0]}" if rows else "no matching EMG row"
                if len(rows) > 1:
                    where += f" (+{len(rows) - 1} more matches)"
                seg = str(int(rv.markers.segment[mrow])) if rv.markers.has_segment else "—"
                entries.append((f"marker:{mrow}", "recorder marker", where, "", seg,
                                rv.markers.labels[mrow], "read-only", "markers.csv", UNKNOWN))
        self.table.setRowCount(len(entries))
        for i, (key, *cols, color) in enumerate(entries):
            for j, v in enumerate(cols):
                it = _item(v, color=color)
                if j == 0:
                    it.setData(Qt.UserRole, key)
                self.table.setItem(i, j, it)


# ------------------------------------------------------------------------------------------

class ComparePanel(QWidget):
    def __init__(self, vm, actions):
        super().__init__()
        self.vm = vm
        lay = QVBoxLayout(self)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        lay.addWidget(self.info)
        row = QHBoxLayout()
        for n in ("store_a", "store_b"):
            b = QToolButton()
            b.setDefaultAction(actions[n])
            b.setToolButtonStyle(Qt.ToolButtonTextOnly)
            row.addWidget(b)
        for k in ("A", "B"):
            b = QPushButton(f"Clear {k}")
            b.clicked.connect(lambda _=False, k=k: vm.set_ab(k, None))
            row.addWidget(b)
        row.addWidget(QLabel("Waveform channel:"))
        self.ch = QComboBox()
        self.ch.addItems(CHANNELS)
        self.ch.currentIndexChanged.connect(self.refresh)
        row.addWidget(self.ch)
        row.addStretch(1)
        lay.addLayout(row)
        self.wave = pg.PlotWidget()
        self.wave.setMinimumHeight(160)
        self.wave.setMenuEnabled(False)
        self.wave.addLegend(offset=(5, 5))
        self.wave.setLabel("bottom", "relative sample offset = sample_index − first sample_index "
                                     "(display coordinate only)")
        lay.addWidget(self.wave, 2)
        self.table = _table(["ch", "A eligible", "A AC RMS", "B eligible", "B AC RMS",
                             "B / A (derived ratio)"], False)
        lay.addWidget(self.table, 2)
        self.bar = pg.PlotWidget()
        self.bar.setMinimumHeight(130)
        self.bar.setMenuEnabled(False)
        self.bar.setMouseEnabled(False, False)
        self.bar.hideButtons()
        self.bar.setTitle("<span style='font-size:9pt'>AC RMS: A (blue) vs B (orange)</span>")
        lay.addWidget(self.bar, 1)
        note = QLabel("<small>Both intervals come from the same recording and use identical "
                      "settings and exclusion policy. Nothing is stretched, interpolated, resampled or "
                      "normalized; index gaps stay gaps. A ratio is a derived comparison of amplitudes, "
                      "not a physiological interpretation.</small>")
        note.setWordWrap(True)
        lay.addWidget(note)
        vm.abChanged.connect(self.refresh)
        vm.reviewChanged.connect(self.refresh)

    def refresh(self):
        vm = self.vm
        emg = vm.emg
        self.wave.clear()
        self.bar.clear()
        self.table.setRowCount(0)
        A, B = vm.ab["A"], vm.ab["B"]

        def desc(k, rows):
            if not rows:
                return f"<b>{k}:</b> not set"
            run = emg.run_of_row(rows[0])
            return (f"<b>{k}:</b> rows [{rows[0]}, {rows[1]}) — {rows[1] - rows[0]:,} rows, "
                    f"last row {rows[1] - 1}, {run.name}")
        if emg is None:
            self.info.setText("")
            return
        self.info.setText(desc("A", A) + "<br>" + desc("B", B) + "<br><small>Store the current "
                          "selection as A (key 1) or B (key 2).</small>")
        if not (A and B):
            return
        res = compare_intervals(emg, A, B)
        c = self.ch.currentIndex()
        for k, color in (("A", A_BRUSH), ("B", B_BRUSH)):
            w = res[k]["wave"]
            y = np.where(w.eligible[:, c], w.values[:, c], np.nan)
            link = np.append(np.diff(w.offset) == 1, False)
            xs, ys, conn = display_series(w.offset.astype(float), y, link, 4000)
            item = pg.PlotCurveItem(xs, ys, connect=conn, pen=pg.mkPen(color[:3], width=1), name=k)
            self.wave.addItem(item)
        self.wave.setTitle(f"<span style='font-size:9pt'>ch{c + 1}: A and B original samples overlaid "
                           "by relative sample offset</span>")
        sa, sb = res["A"]["stats"], res["B"]["stats"]
        self.table.setRowCount(N_CHANNELS)
        for i in range(N_CHANNELS):
            ratio = (sb[i].ac_rms / sa[i].ac_rms if sa[i].ac_rms and sb[i].ac_rms is not None
                     else None)
            vals = [f"ch{i + 1}", f"{sa[i].n_eligible:,}", fmt(sa[i].ac_rms), f"{sb[i].n_eligible:,}",
                    fmt(sb[i].ac_rms), fmt(ratio, 3)]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, _item(v))
        _bars(self.bar, [[s.ac_rms for s in sa], [s.ac_rms for s in sb]], x_offsets=(-0.2, 0.2),
              width=0.38, brushes=[pg.mkBrush(*A_BRUSH[:3]), pg.mkBrush(*B_BRUSH[:3])])


# ------------------------------------------------------------------------------------------

def recording_html(review) -> str:
    emg = review.emg
    h = []
    e = html.escape
    if is_synthetic(review.session_meta):
        h.append("<p style='background:#fff3b0;padding:4px'><b>SYNTHETIC DEMO DATA</b> — generated, "
                 "not a recording. Nothing here describes a real participant or device.</p>")
    h.append("<h3>Sources (immutable snapshots)</h3><table border='0' cellspacing='2'>")
    for s in review.sources.values():
        h.append(f"<tr><td><b>{e(KIND_LABELS.get(s.kind, s.kind))}</b></td><td>{e(s.original_name)}"
                 f"</td><td>{s.size:,} B</td><td>{'' if s.n_rows is None else f'{s.n_rows:,} rows'}</td>"
                 f"<td><code>sha256 {s.sha256[:16]}…</code></td></tr>")
        for n in s.notes:
            h.append(f"<tr><td></td><td colspan='4'><i>{e(n)}</i></td></tr>")
    h.append("</table>")
    if review.folder:
        h.append(f"<p>Saved review: <code>{e(str(review.folder))}</code></p>")
    warn = list(review.warnings) + list(review.import_notes)
    timings = [run_timing(emg, r) for r in emg.runs]
    warn += timing_warnings(timings)
    if warn:
        h.append("<h3>Warnings and notes</h3><ul>" + "".join(f"<li>{e(w)}</li>" for w in warn) + "</ul>")
    fields = session_fields(review.session_meta)
    h.append("<h3>Session metadata</h3>")
    if fields:
        h.append("<table>" + "".join(f"<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>" for k, v in fields)
                 + "</table>")
    else:
        h.append("<p>No session.json imported — device, settings and units are <b>unknown</b>.</p>")

    g = integrity(emg)
    h.append("<h3>File / recording integrity</h3><table>")
    rows = [("data rows", f"{g.rows:,}"), ("segment runs", g.runs),
            ("lost rows (lost≠0)", "column not recorded" if g.lost_rows is None else f"{g.lost_rows:,}"),
            ("lost unknown (empty/absent)", f"{g.lost_unknown:,}"),
            ("rows with valid_mask bits cleared", "column not recorded" if g.mask_rows_partial is None
             else f"{g.mask_rows_partial:,}"),
            ("valid_mask unknown", f"{g.mask_unknown:,}"),
            ("index discontinuities (gaps / duplicates / decreases)",
             f"{g.index_gaps} / {g.index_duplicates} / {g.index_decreases}"),
            ("missing sample indices (in gaps)", f"{g.missing_indices:,}"),
            ("non-finite channel values (incl. empty)", ", ".join(f"{c}:{n}" for c, n in g.nonfinite.items() if n) or "none"),
            ("timestamp columns non-finite", ", ".join(f"{c}:{n}" for c, n in g.time_nonfinite.items()) or "—"),
            ("timestamp decreases within segments", ", ".join(f"{c}:{n}" for c, n in g.time_decreases.items()) or "—"),
            ("extra columns (preserved)", ", ".join(g.extra_columns) or "none"),
            ("blank lines (not data rows)", ", ".join(map(str, g.blank_lines[:10])) or "none")]
    for col, (cnt, ex) in g.unparsed.items():
        rows.append((f"unparsed cells in {col}", f"{cnt} (e.g. " + ", ".join(f"row {r}: {t!r}" for r, t in ex) + ")"))
    h.append("".join(f"<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>" for k, v in rows) + "</table>")

    h.append("<h3>Timing (per segment run)</h3>")
    h.append(f"<p><small>{e(DEVICE_TIME_NOTE)} {e(RECEIPT_NOTE)} Estimator: {e(RATE_ESTIMATOR)} "
             "These figures describe delivery to the host; they do not verify the physical "
             "sampling rate or sensor-to-host latency.</small></p>")
    for k, v in reported_emg_rates(review.session_meta):
        h.append(f"<div>{e(k)}: <b>{e(str(v))}</b> (not verified)</div>")
    for t in timings:
        h.append(f"<p><b>{e(t.run.name)}</b> — {t.n_rows:,} rows, sample_index {t.index_first}…{t.index_last}"
                 f" ({t.missing_indices:,} missing in {t.gaps} gap(s), {t.duplicates} duplicates)<br>")
        if t.receipt_clock:
            h.append(f"receipt ({t.receipt_clock}): span {fmt(t.receipt_span_s, 6)} s, "
                     f"{fmt(t.receipt_groups)} receipt groups (rows/group median {fmt(t.rows_per_group_median)}, "
                     f"max {fmt(t.rows_per_group_max)}); gap between groups median "
                     f"{fmt(t.receipt_gap_ms_median)} ms, p95 {fmt(t.receipt_gap_ms_p95)} ms, max "
                     f"{fmt(t.receipt_gap_ms_max)} ms<br>")
        h.append("received throughput: " + (f"<b>{t.received_rows_per_s:.1f} rows/s</b>"
                                            if t.received_rows_per_s else e(t.rate_note)) + "<br>")
        if t.nominal_span_s is not None:
            h.append(f"SDK nominal elapsed (device_time): {fmt(t.nominal_span_s, 6)} s; index steps per "
                     f"nominal second: {fmt(t.nominal_index_rate, 6)}")
            if t.nominal_vs_receipt:
                h.append(f"; nominal/receipt elapsed ratio {t.nominal_vs_receipt:.3f}"
                         + (" <b style='color:#b00'>(mismatch — acquisition timing unverified)</b>"
                            if t.mismatch else ""))
        if t.callback_ids is not None:
            h.append(f"<br>explicit callback_id values: {t.callback_ids:,}")
        h.append("</p>")
    if review.contact is not None:
        cs = contact_summary(review.contact)
        h.append("<h3>Electrode–skin impedance observations</h3>")
        h.append(f"<p>{cs.observations} logged observations"
                 + (f" per segment {cs.per_segment}" if cs.per_segment else "")
                 + (f"; {cs.receipt_clock} span {fmt(cs.receipt_span_s)} s" if cs.receipt_span_s else "")
                 + f"<br><small>{e(cs.note)} Values are recorded in Ω and shown in kΩ.</small></p><table>"
                 "<tr><th>ch</th><th>usable</th><th>unknown</th><th>min kΩ</th><th>median kΩ</th><th>max kΩ</th></tr>")
        for c in range(N_CHANNELS):
            h.append(f"<tr><td>imp{c + 1}</td><td>{cs.usable[c]}</td><td>{cs.unknown[c]}</td>"
                     f"<td>{fmt(cs.kohm_min[c])}</td><td>{fmt(cs.kohm_median[c])}</td><td>{fmt(cs.kohm_max[c])}</td></tr>")
        h.append("</table>")
    h.append("<h3>Derived views</h3><p><small>" + e(ENVELOPE_DEFINITION) + "</small></p>")
    h.append("<p><small>No universal signal-quality score is shown. Importing a file does not show "
             "physiological quality or synchronization with any other recording.</small></p>")
    return "".join(h)


class RecordingPanel(QWidget):
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        lay = QVBoxLayout(self)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(False)
        lay.addWidget(self.text)
        vm.reviewChanged.connect(self.refresh)
        vm.companionsChanged.connect(self.refresh)

    def refresh(self):
        rv = self.vm.review
        self.text.setHtml(recording_html(rv) if rv else "")


class Inspector(QTabWidget):
    def __init__(self, vm, actions):
        super().__init__()
        self.sample = SamplePanel(vm)
        self.selection = SelectionPanel(vm, actions)
        self.annotations = AnnotationsPanel(vm, actions)
        self.compare = ComparePanel(vm, actions)
        self.recording = RecordingPanel(vm)
        self.addTab(self.sample, "Sample")
        self.addTab(self.selection, "Selection")
        self.addTab(self.annotations, "Annotations")
        self.addTab(self.compare, "Compare A/B")
        self.addTab(self.recording, "Recording")
