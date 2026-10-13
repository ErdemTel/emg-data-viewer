"""Small dialogs: go to sample, annotation editor, help."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QGridLayout, QHBoxLayout, QInputDialog, QLabel,
                             QLineEdit, QMessageBox, QPlainTextEdit, QRadioButton, QSpinBox,
                             QTextBrowser, QVBoxLayout, QWidget)

from ..annotations import REVIEW_STATUSES, TYPE_BOOKMARK
from ..schema import N_CHANNELS


class GoToDialog(QDialog):
    """Go to an original data row, or to a recorded sample_index (optionally in a segment)."""

    def __init__(self, emg, current_row, parent=None):
        super().__init__(parent)
        self.emg = emg
        self.setWindowTitle("Go to sample")
        lay = QVBoxLayout(self)
        grid = QGridLayout()
        self.by_row = QRadioButton("Original data row (0-based)")
        self.by_index = QRadioButton("Recorded sample_index")
        grp = QButtonGroup(self)
        grp.addButton(self.by_row)
        grp.addButton(self.by_index)
        self.by_row.setChecked(True)
        self.row = QSpinBox()
        self.row.setRange(0, emg.n - 1)
        self.row.setValue(current_row or 0)
        self.index = QLineEdit()
        self.index.setPlaceholderText("e.g. 632950")
        self.segment = QComboBox()
        self.segment.addItem("any segment", None)
        if emg.has_segment:
            for s in sorted({r.segment for r in emg.runs}):
                self.segment.addItem(f"segment {s}", s)
        grid.addWidget(self.by_row, 0, 0)
        grid.addWidget(self.row, 0, 1)
        grid.addWidget(self.by_index, 1, 0)
        grid.addWidget(self.index, 1, 1)
        grid.addWidget(QLabel("in"), 2, 0, alignment=Qt.AlignRight)
        grid.addWidget(self.segment, 2, 1)
        lay.addLayout(grid)
        self.index.textEdited.connect(lambda: self.by_index.setChecked(True))
        self.row.valueChanged.connect(lambda: self.by_row.setChecked(True))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def resolve(self):
        """The chosen row, asking explicitly when a sample index matches several rows."""
        if self.by_row.isChecked():
            return self.row.value()
        try:
            idx = int(self.index.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Go to sample", "Enter an integer sample_index.")
            return None
        rows = self.emg.find_rows(idx, self.segment.currentData())
        if not rows:
            QMessageBox.information(self, "Go to sample", f"No row has sample_index {idx}"
                                    + ("" if self.segment.currentData() is None
                                       else f" in segment {self.segment.currentData()}") + ".")
            return None
        if len(rows) == 1:
            return rows[0]
        items = []
        for r in rows[:500]:
            raw = self.emg.raw_row(r)
            run = self.emg.run_of_row(r)
            t = raw.get("host_mono") or raw.get("host_time") or ""
            items.append(f"row {r} — {run.name}, sample_index {raw['sample_index']}"
                         + (f", receipt {t}" if t else ""))
        choice, ok = QInputDialog.getItem(
            self, "Ambiguous sample_index",
            f"sample_index {idx} occurs in {len(rows)} rows. Choose one:", items, 0, False)
        if not ok:
            return None
        return rows[items.index(choice)]


class AnnotationDialog(QDialog):
    def __init__(self, title, rows_text, labels_seen, annotation=None, is_bookmark=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        lay = QVBoxLayout(self)
        info = QLabel(rows_text)
        info.setWordWrap(True)
        lay.addWidget(info)
        form = QFormLayout()
        self.label = QComboBox()
        self.label.setEditable(True)
        self.label.addItems([""] + sorted(set(labels_seen)))
        self.label.setToolTip("Free text, e.g. 'pre-grasp candidate'. A label is a user "
                              "assessment, not a verified event.")
        self.status = QComboBox()
        self.status.addItems(REVIEW_STATUSES)
        self.notes = QPlainTextEdit()
        self.notes.setFixedHeight(80)
        form.addRow("Label", self.label)
        form.addRow("Review status", self.status)
        form.addRow("Notes", self.notes)
        chw = QWidget()
        chl = QHBoxLayout(chw)
        chl.setContentsMargins(0, 0, 0, 0)
        self.channels = []
        for c in range(N_CHANNELS):
            cb = QCheckBox(f"ch{c + 1}")
            cb.setChecked(True)
            chl.addWidget(cb)
            self.channels.append(cb)
        form.addRow("Channels", chw)
        lay.addLayout(form)
        if annotation is not None:
            self.label.setEditText(annotation.label)
            self.status.setCurrentText(annotation.review_status)
            self.notes.setPlainText(annotation.notes)
            sel = set(annotation.channel_list())
            for c, cb in enumerate(self.channels):
                cb.setChecked(c in sel)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _ok(self):
        if not self.selected_channels():
            QMessageBox.warning(self, self.windowTitle(), "Select at least one channel.")
            return
        self.accept()

    def selected_channels(self):
        return [c for c, cb in enumerate(self.channels) if cb.isChecked()]

    def values(self):
        return {"label": self.label.currentText().strip(), "notes": self.notes.toPlainText(),
                "review_status": self.status.currentText(), "channels": self.selected_channels()}


HELP_HTML = """
<h3>EMG Review — quick reference</h3>
<p><b>Workflow:</b> drop an EMG recording (file or folder) → explore → click to inspect exact rows →
select / compare / label intervals → Save Review.</p>
<table cellspacing="4">
<tr><td><b>Mouse wheel</b></td><td>zoom the time axis around the pointer</td></tr>
<tr><td><b>Drag</b></td><td>pan horizontally</td></tr>
<tr><td><b>Shift + drag</b>, or <b>S</b> (Select mode) + drag</td><td>select original rows</td></tr>
<tr><td><b>Click</b></td><td>inspect the nearest original row</td></tr>
<tr><td><b>← / →</b></td><td>previous / next ORIGINAL row (Shift: 10 rows, Ctrl: 100 rows)</td></tr>
<tr><td><b>Home / End</b></td><td>first / last row of the segment</td></tr>
<tr><td><b>[ / ]</b></td><td>set selection start / inclusive end at the cursor</td></tr>
<tr><td><b>Esc</b></td><td>clear selection</td></tr>
<tr><td><b>Z / F</b></td><td>zoom to selection / fit whole segment</td></tr>
<tr><td><b>+ / −</b>, <b>PgUp / PgDn</b></td><td>zoom in / out, pan by most of a screen</td></tr>
<tr><td><b>C</b></td><td>centre the view on the cursor</td></tr>
<tr><td><b>L</b> / <b>B</b></td><td>label the selection / add a bookmark at the cursor</td></tr>
<tr><td><b>1 / 2</b></td><td>store the selection as A / B for comparison</td></tr>
<tr><td><b>Ctrl+G</b></td><td>go to an original row or a sample_index (ambiguity is resolved explicitly)</td></tr>
<tr><td><b>Ctrl+O / Ctrl+Shift+O / Ctrl+S / Ctrl+E</b></td><td>import / open review / save review / export annotations</td></tr>
</table>
<h4>What the views mean</h4>
<ul>
<li><b>Axis modes.</b> <i>Sample index</i> (default) and <i>original row</i> need no timing assumption.
<i>Host receipt time</i> is when the host received each batch (rows of a batch share it).
<i>SDK nominal time</i> is the SDK's device_time — a nominal timeline, <b>acquisition timing unverified</b>.</li>
<li><b>Equal scale</b> (default): every channel uses the same vertical range. Independent scales are
labelled and make amplitudes non-comparable between channels.</li>
<li>Lines are never drawn across segments, sample-index gaps or excluded samples. Orange dashed lines mark
index discontinuities. Wide views are drawn as per-pixel min/max for display only.</li>
<li>The <b>derived envelope</b> (optional) is a trailing AC RMS over a SAMPLE COUNT; it restarts at gaps
and is not a filtered or replaced signal.</li>
<li><b>Electrode–skin impedance</b> (contact.csv) is plotted as logged observations. The cursor shows the
last observation logged in the same segment at or before the row's receipt time; its age is time since
it was LOGGED, not the age of the measurement.</li>
<li>Annotations use half-open original row ranges [start, end); the inclusive last row is end − 1.
No cross-modal (camera/robot) alignment is performed.</li>
</ul>
"""


def show_help(parent):
    d = QDialog(parent)
    d.setWindowTitle("EMG Review — help")
    d.resize(720, 640)
    lay = QVBoxLayout(d)
    t = QTextBrowser()
    t.setHtml(HELP_HTML)
    lay.addWidget(t)
    bb = QDialogButtonBox(QDialogButtonBox.Close)
    bb.rejected.connect(d.reject)
    bb.accepted.connect(d.accept)
    lay.addWidget(bb)
    d.exec_()
