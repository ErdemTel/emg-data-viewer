"""Main window: welcome screen, toolbar, import/open/save flows, shortcuts."""

import atexit
import signal
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QProgressDialog, QPushButton, QSpinBox, QSplitter, QStackedWidget,
                             QToolBar, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..annotations import (SOURCE_DEMO, TYPE_BOOKMARK, TYPE_INTERVAL, SelectionError,
                           channels_text, export_annotations, make_annotation)
from ..demo import demo_annotation_rows, write_demo
from ..diagnostics import is_synthetic, run_timing, timing_warnings
from ..schema import KIND_EMG, KIND_LABELS, N_CHANNELS
from ..snapshot import SnapshotError, WorkArea
from ..workspace import (Review, WorkspaceError, import_issues, import_paths, open_review,
                         save_review)
from .dialogs import AnnotationDialog, GoToDialog, show_help
from .inspector import Inspector
from .plots import CH_COLORS, EmgPlotArea
from .viewmodel import ViewModel
from .worker import Task


def _demo_job(folder, workarea):
    write_demo(folder)
    return import_paths([folder], workarea)


class WelcomePage(QWidget):
    def __init__(self, win):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addStretch(1)
        box = QFrame()
        box.setObjectName("dropzone")
        box.setStyleSheet("#dropzone { border: 2px dashed #8a8a8a; border-radius: 12px; }")
        bl = QVBoxLayout(box)
        bl.setContentsMargins(40, 40, 40, 40)
        title = QLabel("Drop an EMG recording here")
        f = title.font()
        f.setPointSize(f.pointSize() + 8)
        title.setFont(f)
        title.setAlignment(Qt.AlignCenter)
        bl.addWidget(title)
        sub = QLabel("an emg.csv, several companion files (contact / imu / markers / session.json), "
                     "or a recording folder.\nFiles are copied into the review as unchanged "
                     "snapshots; your originals are never modified.")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #555;")
        bl.addWidget(sub)
        row = QHBoxLayout()
        row.addStretch(1)
        for text, slot in (("Browse Files…", win.browse_files), ("Import Folder…", win.browse_folder),
                           ("Open Review…", win.browse_review)):
            b = QPushButton(text)
            b.setMinimumHeight(34)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        bl.addLayout(row)
        demo = QPushButton("Open Demo (SYNTHETIC data — not a recording)")
        demo.setStyleSheet("color: #7a5a00;")
        demo.clicked.connect(win.open_demo)
        dr = QHBoxLayout()
        dr.addStretch(1)
        dr.addWidget(demo)
        dr.addStretch(1)
        bl.addLayout(dr)
        outer = QHBoxLayout()
        outer.addStretch(1)
        outer.addWidget(box, 3)
        outer.addStretch(1)
        lay.addLayout(outer)
        lay.addStretch(2)


class ReviewPage(QWidget):
    def __init__(self, win, vm, actions):
        super().__init__()
        self.vm = vm
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.hide()
        lay.addWidget(self.banner)
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Channels:"))
        self.ch_boxes = []
        for c in range(N_CHANNELS):
            cb = QCheckBox(f"ch{c + 1}")
            cb.setChecked(True)
            cb.setStyleSheet(f"color: {CH_COLORS[c]}; font-weight: bold;")
            cb.setToolTip("Show/hide this channel's plot (data are never removed)")
            cb.toggled.connect(lambda on, c=c: vm.set_channel_visible(c, on))
            ctl.addWidget(cb)
            self.ch_boxes.append(cb)
        ctl.addSpacing(12)
        self.ymode = QComboBox()
        self.ymode.addItem("Equal vertical scale (all channels)", "equal")
        self.ymode.addItem("Independent scales (NOT comparable)", "independent")
        self.ymode.currentIndexChanged.connect(lambda i: vm.set_y_mode(self.ymode.itemData(i)))
        ctl.addWidget(self.ymode)
        ctl.addSpacing(12)
        self.env = QCheckBox("Derived envelope")
        self.env.setToolTip("Overlay a derived activity envelope: trailing AC RMS over a SAMPLE-COUNT "
                            "window (display only; restarts at gaps; original waveform stays visible)")
        self.env_w = QSpinBox()
        self.env_w.setRange(2, 100000)
        self.env_w.setValue(128)
        self.env_w.setSuffix(" samples")
        self.env.toggled.connect(self._env)
        self.env_w.editingFinished.connect(self._env)
        ctl.addWidget(self.env)
        ctl.addWidget(self.env_w)
        ctl.addSpacing(12)
        self.comp = QCheckBox("Companion tracks")
        self.comp.setChecked(True)
        self.comp.toggled.connect(self._comp)
        ctl.addWidget(self.comp)
        self.pol_lbl = QLabel("Impedance display policy ≤")
        self.pol_lo = QDoubleSpinBox()
        self.pol_hi = QDoubleSpinBox()
        for sp, v in ((self.pol_lo, 500.0), (self.pol_hi, 999.0)):
            sp.setRange(1, 1e6)
            sp.setDecimals(0)
            sp.setValue(v)
            sp.setSuffix(" kΩ")
            sp.setToolTip("Configurable DISPLAY policy for impedance colours (recorder convention "
                          "500/999 kΩ) — not a universal signal-validity rule")
            sp.editingFinished.connect(self._policy)
        ctl.addWidget(self.pol_lbl)
        ctl.addWidget(self.pol_lo)
        ctl.addWidget(QLabel("/"))
        ctl.addWidget(self.pol_hi)
        ctl.addStretch(1)
        lay.addLayout(ctl)
        self.split = QSplitter(Qt.Horizontal)
        self.plots = EmgPlotArea(vm)
        self.inspector = Inspector(vm, actions)
        self.split.addWidget(self.plots)
        self.split.addWidget(self.inspector)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        self.split.setSizes([900, 560])
        lay.addWidget(self.split, 1)
        vm.reviewChanged.connect(self.sync_controls)

    def _env(self):
        self.vm.set_envelope(self.env.isChecked(), self.env_w.value())

    def _comp(self, on):
        self.vm.show_companions = on
        self.vm.companionsChanged.emit()

    def _policy(self):
        lo, hi = self.pol_lo.value(), self.pol_hi.value()
        if hi < lo:
            hi = lo
            self.pol_hi.setValue(hi)
        self.vm.contact_policy_kohm = (lo, hi)
        self.vm.companionsChanged.emit()
        self.vm.displayChanged.emit()

    def sync_controls(self):
        vm = self.vm
        if vm.review is None:
            return
        for w in [self.ymode, self.env, self.env_w, self.comp, self.pol_lo, self.pol_hi] + self.ch_boxes:
            w.blockSignals(True)
        for c, cb in enumerate(self.ch_boxes):
            cb.setChecked(vm.channels_visible[c])
        self.ymode.setCurrentIndex(0 if vm.y_mode == "equal" else 1)
        self.env.setChecked(vm.envelope_on)
        self.env_w.setValue(vm.envelope_window)
        self.comp.setChecked(vm.show_companions)
        self.pol_lo.setValue(vm.contact_policy_kohm[0])
        self.pol_hi.setValue(vm.contact_policy_kohm[1])
        for w in [self.ymode, self.env, self.env_w, self.comp, self.pol_lo, self.pol_hi] + self.ch_boxes:
            w.blockSignals(False)
        has_contact = vm.review.contact is not None
        for w in (self.pol_lbl, self.pol_lo, self.pol_hi):
            w.setVisible(has_contact)
        self.comp.setVisible(vm.review.contact is not None or vm.review.imu is not None)
        self.update_banner()

    def update_banner(self):
        rv = self.vm.review
        parts = []
        style = "padding: 4px; border-radius: 4px;"
        if is_synthetic(rv.session_meta):
            parts.append("<b>SYNTHETIC DEMO DATA</b> — generated by the app, not a recording.")
            style += "background: #fff3b0;"
        warns = timing_warnings([run_timing(rv.emg, r) for r in rv.emg.runs])
        if warns:
            parts.append(f"<b>Timing:</b> SDK nominal and host receipt elapsed times disagree in "
                         f"{len(warns)} segment(s) — acquisition timing unverified; timestamps are shown "
                         "as recorded (see Recording tab).")
            if "background" not in style:
                style += "background: #ffe3e3;"
        self.banner.setStyleSheet(style)
        self.banner.setText("<br>".join(parts))
        self.banner.setVisible(bool(parts))


class MainWindow(QMainWindow):
    def __init__(self, workarea: WorkArea = None):
        super().__init__()
        self.workarea = workarea or WorkArea()
        self.vm = ViewModel()
        self._tasks = set()
        self.setAcceptDrops(True)
        self.resize(1500, 950)
        self.a = self._make_actions()
        self._make_toolbar()
        self.welcome = WelcomePage(self)
        self.page = ReviewPage(self, self.vm, self.a)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.welcome)
        self.stack.addWidget(self.page)
        self.setCentralWidget(self.stack)
        self.statusBar().showMessage("Drag: pan · Wheel: zoom · Shift+drag or S: select · Click: inspect · "
                                     "←/→: original rows · F1: help")
        self.vm.message.connect(lambda m: self.statusBar().showMessage(m, 8000))
        self.vm.dirtyChanged.connect(self._update_title)
        self.vm.annotationsChanged.connect(self.vm.mark_dirty)
        self.vm.reviewChanged.connect(self._on_review)
        self.vm.runChanged.connect(self._sync_run_combo)
        self.vm.cursorChanged.connect(self._cursor_status)
        self.page.inspector.annotations.jumpRequested.connect(self.jump_to)
        self.page.inspector.annotations.editRequested.connect(self.edit_annotation)
        self.page.inspector.annotations.deleteRequested.connect(self.delete_annotation)
        self._enable_review_actions(False)
        self._update_title()

    # ------------------------------------------------------------------------------------
    # actions / toolbar
    # ------------------------------------------------------------------------------------
    def _make_actions(self):
        spec = [
            ("import_files", "Import / Add Files…", "Ctrl+O", self.browse_files,
             "Import an EMG recording, or add companion files to the current review"),
            ("import_folder", "Import Folder…", None, self.browse_folder,
             "Import the recognised files directly inside one folder (no subfolders)"),
            ("open_review", "Open Review…", "Ctrl+Shift+O", self.browse_review, "Open a saved review folder"),
            ("save", "Save Review", "Ctrl+S", self.save, "Save annotations and view settings with "
             "unchanged copies of the source files"),
            ("save_as", "Save Review As…", None, self.save_as, None),
            ("export", "Export Annotations…", "Ctrl+E", self.export, "Write user annotations to a CSV file"),
            ("demo", "Open Demo (SYNTHETIC)", None, self.open_demo, "Generated demo data, not a recording"),
            ("fit", "Fit Recording", "F", lambda: self.plots.fit(), "Show the whole segment (F)"),
            ("zoom_sel", "Zoom to Selection", "Z", self.zoom_selection, "Z"),
            ("goto", "Go to Sample…", "Ctrl+G", self.goto, "Go to an original row or sample_index"),
            ("select_mode", "Select mode", "S", self._toggle_select, "Plain drag selects rows instead of "
             "panning (Shift+drag always selects)"),
            ("set_start", "Set Start [", "[", self.set_start, "Selection starts at the cursor row"),
            ("set_end", "Set End ]", "]", self.set_end, "Cursor row becomes the INCLUSIVE last selected row"),
            ("clear_sel", "Clear (Esc)", "Esc", lambda: self.vm.set_selection(None), "Clear the selection"),
            ("label_sel", "Label Selection… (L)", "L", self.label_selection,
             "Create an interval annotation from the selection"),
            ("bookmark", "Add Bookmark (B)", "B", self.add_bookmark, "Bookmark the cursor row"),
            ("store_a", "Store as A (1)", "1", lambda: self.store("A"), "Store the selection as A"),
            ("store_b", "Store as B (2)", "2", lambda: self.store("B"), "Store the selection as B"),
            ("prev", "Previous row", "Left", lambda: self.step(-1), None),
            ("next", "Next row", "Right", lambda: self.step(1), None),
            ("prev10", "Back 10 rows", "Shift+Left", lambda: self.step(-10), None),
            ("next10", "Forward 10 rows", "Shift+Right", lambda: self.step(10), None),
            ("prev100", "Back 100 rows", "Ctrl+Left", lambda: self.step(-100), None),
            ("next100", "Forward 100 rows", "Ctrl+Right", lambda: self.step(100), None),
            ("home", "First row of segment", "Home", lambda: self.step(-10 ** 12), None),
            ("end", "Last row of segment", "End", lambda: self.step(10 ** 12), None),
            ("zoom_in", "Zoom in", "+", lambda: self.plots.zoom_factor(0.5), None),
            ("zoom_out", "Zoom out", "-", lambda: self.plots.zoom_factor(2.0), None),
            ("pan_left", "Pan left", "PgUp", lambda: self.plots.pan_fraction(-0.8), None),
            ("pan_right", "Pan right", "PgDown", lambda: self.plots.pan_fraction(0.8), None),
            ("center", "Centre on cursor", "C", self.center_cursor, None),
            ("help", "Help", "F1", lambda: show_help(self), "Shortcuts and what the views mean"),
            ("quit", "Quit", "Ctrl+Q", self.close, None),
        ]
        acts = {}
        for key, text, sc, slot, tip in spec:
            a = QAction(text, self)
            if sc:
                a.setShortcut(QKeySequence(sc))
                if key == "zoom_in":
                    a.setShortcuts([QKeySequence("+"), QKeySequence("=")])
            a.setShortcutContext(Qt.WindowShortcut)
            if tip:
                a.setToolTip(f"{tip}" + (f"  [{sc}]" if sc else ""))
            a.triggered.connect(slot)
            self.addAction(a)
            acts[key] = a
        acts["select_mode"].setCheckable(True)
        m = self.menuBar().addMenu("&File")
        for k in ("import_files", "import_folder", "open_review", None, "save", "save_as", "export",
                  None, "demo", None, "quit"):
            m.addSeparator() if k is None else m.addAction(acts[k])
        m = self.menuBar().addMenu("&View")
        for k in ("fit", "zoom_sel", "goto", "center", "zoom_in", "zoom_out", None, "select_mode"):
            m.addSeparator() if k is None else m.addAction(acts[k])
        m = self.menuBar().addMenu("&Annotate")
        for k in ("set_start", "set_end", "clear_sel", None, "label_sel", "bookmark", None,
                  "store_a", "store_b"):
            m.addSeparator() if k is None else m.addAction(acts[k])
        self.menuBar().addMenu("&Help").addAction(acts["help"])
        return acts

    def _make_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        for k in ("import_files", "open_review", "save", "export"):
            tb.addAction(self.a[k])
        tb.addSeparator()
        self.rec_label = QLabel("  no recording  ")
        self.rec_label.setStyleSheet("font-weight: bold;")
        tb.addWidget(self.rec_label)
        tb.addSeparator()
        tb.addWidget(QLabel(" Axis: "))
        self.axis_combo = QComboBox()
        self.axis_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.axis_combo.setMinimumContentsLength(24)
        self.axis_combo.currentIndexChanged.connect(self._axis_selected)
        tb.addWidget(self.axis_combo)
        tb.addWidget(QLabel(" Segment: "))
        self.run_combo = QComboBox()
        self.run_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.run_combo.setMinimumContentsLength(30)
        self.run_combo.currentIndexChanged.connect(self._run_selected)
        tb.addWidget(self.run_combo)
        tb.addSeparator()
        for k in ("fit", "zoom_sel", "goto", "select_mode", "help"):
            tb.addAction(self.a[k])

    @property
    def plots(self):
        return self.page.plots

    def _enable_review_actions(self, on):
        for k, a in self.a.items():
            if k not in ("import_files", "import_folder", "open_review", "demo", "help", "quit"):
                a.setEnabled(on)
        self.axis_combo.setEnabled(on)
        self.run_combo.setEnabled(on)

    # ------------------------------------------------------------------------------------
    # review lifecycle
    # ------------------------------------------------------------------------------------
    def _update_title(self):
        rv = self.vm.review
        if rv is None:
            self.setWindowTitle(f"{APP_NAME} {__version__}")
            return
        demo = " [SYNTHETIC DEMO]" if is_synthetic(rv.session_meta) else ""
        self.setWindowTitle(f"{APP_NAME} — {rv.emg.name}{demo}{' *' if rv.dirty else ''}")
        self.rec_label.setText(f"  {rv.emg.name}{demo} · {rv.emg.n:,} rows  ")

    def _on_review(self):
        rv = self.vm.review
        self._enable_review_actions(rv is not None)
        self.stack.setCurrentWidget(self.page if rv else self.welcome)
        if rv is None:
            return
        self.axis_combo.blockSignals(True)
        self.axis_combo.clear()
        for m in rv.emg.available_axes():
            self.axis_combo.addItem(rv.emg.axis_short(m), m)
            self.axis_combo.setItemData(self.axis_combo.count() - 1, rv.emg.axis_title(m), Qt.ToolTipRole)
        self.axis_combo.setCurrentIndex(self.axis_combo.findData(self.vm.axis))
        self.axis_combo.blockSignals(False)
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for r in rv.emg.runs:
            self.run_combo.addItem(r.label())
        self.run_combo.setCurrentIndex(self.vm.run_index)
        self.run_combo.blockSignals(False)
        self.a["select_mode"].setChecked(self.vm.select_mode)
        if self.vm.cursor_row is None:
            self.vm.set_cursor(self.vm.run.start)
        self._update_title()

    def _sync_run_combo(self):
        self.run_combo.blockSignals(True)
        self.run_combo.setCurrentIndex(self.vm.run_index)
        self.run_combo.blockSignals(False)

    def _axis_selected(self, i):
        if i >= 0:
            self.vm.set_axis(self.axis_combo.itemData(i))
            self.plots.update_cursor()

    def _run_selected(self, i):
        if i >= 0:
            self.vm.set_run(i)

    def _cursor_status(self):
        vm = self.vm
        if vm.emg is None or vm.cursor_row is None:
            return
        r = vm.cursor_row
        self.statusBar().showMessage(f"Cursor: original row {r} · sample_index "
                                     f"{vm.emg.raw_row(r)['sample_index']} · {vm.run.name}")

    def _set_review(self, review):
        old = self.vm.review
        self.vm.set_review(review)
        if old is not None:
            for s in old.sources.values():
                if self.workarea.owns(s.snapshot_path):
                    self.workarea.discard(s.snapshot_path.parent)

    def _confirm_discard(self) -> bool:
        rv = self.vm.review
        if rv is None or not rv.dirty:
            return True
        r = QMessageBox.question(
            self, "Unsaved changes",
            "The current review has unsaved annotations or changes. Discard them?",
            QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        return r == QMessageBox.Discard

    # ------------------------------------------------------------------------------------
    # background tasks
    # ------------------------------------------------------------------------------------
    def _run_task(self, label, fn, args, on_done):
        t = Task(fn, *args, parent=self)
        dlg = QProgressDialog(label, None, 0, 0, self)
        dlg.setWindowTitle(APP_NAME)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(300)
        dlg.setCancelButton(None)
        dlg.setValue(0)

        def ok(result):
            dlg.close()
            on_done(result)

        def bad(msg, tb):
            dlg.close()
            self._error(f"{label.rstrip('…')} failed", msg, tb)

        t.succeeded.connect(ok)
        t.failed.connect(bad)
        t.finished.connect(lambda: (self._tasks.discard(t), t.deleteLater(), dlg.deleteLater()))
        self._tasks.add(t)
        t.start()
        return t

    def busy(self) -> bool:
        return bool(self._tasks)

    def _error(self, title, text, details=""):
        box = QMessageBox(QMessageBox.Warning, title, text, QMessageBox.Ok, self)
        if details:
            box.setDetailedText(details)
        box.exec_()

    # ------------------------------------------------------------------------------------
    # import
    # ------------------------------------------------------------------------------------
    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Import EMG recording / companion files", "",
                                                "Recording files (*.csv *.json);;All files (*)")
        if files:
            self.import_paths(files)

    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Import a recording folder")
        if d:
            self.import_paths([d])

    def import_paths(self, paths):
        paths = [str(p) for p in paths if p]
        if paths and not self.busy():
            self._run_task("Importing (copying and checking files)…", import_paths,
                           (paths, self.workarea), self._import_done)

    @staticmethod
    def _outcome_details(out) -> str:
        lines = [f"imported: {s.original_name} → {KIND_LABELS.get(s.kind, s.kind)} "
                 f"(sha256 {s.sha256[:12]}…)" for s in out.sources]
        lines += [f"not imported: {n} — {why}" for n, why in out.rejected]
        lines += [f"note: {n}" for n in out.notes]
        return "\n".join(lines)

    def _ask_issues(self, issues) -> bool:
        box = QMessageBox(QMessageBox.Warning, "Files may not belong together",
                          "These files look inconsistent with each other or with the loaded recording:"
                          "\n\n• " + "\n• ".join(issues) + "\n\nNothing is merged automatically. "
                          "Import anyway?", QMessageBox.Yes | QMessageBox.No, self)
        box.setDefaultButton(QMessageBox.No)
        return box.exec_() == QMessageBox.Yes

    def _import_done(self, out):
        details = self._outcome_details(out)
        if out.review_folders and not out.sources and not out.errors:
            r = QMessageBox.question(self, "Saved review", f"{out.review_folders[0]} is a saved EMG "
                                     "Review workspace. Open it?")
            if r == QMessageBox.Yes:
                self.open_review_path(out.review_folders[0])
            return
        if out.errors:
            out.discard(self.workarea)
            self._error("Import failed", "\n\n".join(out.errors), details)
            return
        rv = self.vm.review
        if not out.of_kind(KIND_EMG):
            if rv is None:
                out.discard(self.workarea)
                self._error("No EMG recording found",
                            "None of the supplied files is a recognised EMG recording (needs "
                            "sample_index and ch1..ch8 columns). Companion files can be added after "
                            "an EMG recording is open.", details)
                return
            if not out.sources:
                out.discard(self.workarea)
                self._error("Nothing imported", "No recognised files were found.", details)
                return
            conflicts = rv.companion_conflicts(out)
            if conflicts:
                out.discard(self.workarea)
                self._error("Not added", "\n".join(conflicts) + "\n\nEMG Review v1 keeps one file "
                            "of each kind per review and never merges sessions.", details)
                return
            issues = import_issues(out, rv)
            if issues and not self._ask_issues(issues):
                out.discard(self.workarea)
                return
            rv.add_companions(out)
            rv.import_notes = rv.import_notes + issues + [
                f"not imported: {n} — {w}" for n, w in out.rejected]
            self.page.sync_controls()
            self.vm.companionsChanged.emit()
            self.vm.dirtyChanged.emit()
            self.statusBar().showMessage("Added: " + ", ".join(s.original_name for s in out.sources), 8000)
            if out.rejected:
                self._error("Some files were not imported", details)
            return
        issues = import_issues(out)
        if issues and not self._ask_issues(issues):
            out.discard(self.workarea)
            return
        if not self._confirm_discard():
            out.discard(self.workarea)
            return
        review = Review.from_import(out)
        review.import_notes = issues + [f"not imported: {n} — {w}" for n, w in out.rejected] + out.notes
        self._set_review(review)
        self.statusBar().showMessage(f"Imported {review.emg.name}: {review.emg.n:,} rows, "
                                     f"{len(review.sources)} file(s)", 8000)
        if out.rejected:
            self._error("Some files were not imported", "The recording was imported, but some "
                        "supplied files were not recognised or were rejected.", details)

    # ------------------------------------------------------------------------------------
    # demo
    # ------------------------------------------------------------------------------------
    def open_demo(self):
        if self.busy() or not self._confirm_discard():
            return
        folder = self.workarea.scratch_dir("demo-")
        self._run_task("Generating SYNTHETIC demo data…", _demo_job, (folder, self.workarea),
                       self._demo_done)

    def _demo_done(self, out):
        if out.errors:
            out.discard(self.workarea)
            self._error("Demo failed", "\n".join(out.errors))
            return
        review = Review.from_import(out)
        for s, e, kind, label in demo_annotation_rows(review.emg.n):
            review.annotations.add(make_annotation(review.emg, s, e, annotation_type=kind, label=label,
                                                   annotation_source=SOURCE_DEMO,
                                                   review_status="candidate"))
        review.import_notes = ["SYNTHETIC demo generated by emg_review.demo (deterministic)."]
        review.mark_clean()
        self._set_review(review)

    # ------------------------------------------------------------------------------------
    # open / save / export
    # ------------------------------------------------------------------------------------
    def browse_review(self):
        d = QFileDialog.getExistingDirectory(self, "Open a saved review folder")
        if d:
            self.open_review_path(d)

    def open_review_path(self, folder):
        if self.busy() or not self._confirm_discard():
            return
        self._run_task("Opening review (verifying source hashes)…", open_review, (folder,),
                       self._open_done)

    def _open_done(self, review):
        review.import_notes = []
        self._set_review(review)
        self.statusBar().showMessage(f"Opened review {review.folder}", 8000)
        if review.warnings:
            self._error("Review opened with warnings", "\n".join(review.warnings))

    def save(self):
        rv = self.vm.review
        if rv is None:
            return
        if rv.folder:
            self._save_to(rv.folder)
        else:
            self.save_as()

    def save_as(self):
        rv = self.vm.review
        if rv is None:
            return
        stem = Path(rv.emg.name).stem
        default = str(Path.home() / f"{stem}_review")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save review as a NEW folder (or an empty one)", default, "Review folder (*)",
            options=QFileDialog.DontConfirmOverwrite)
        if path:
            self._save_to(Path(path))

    def _save_to(self, folder):
        if self.busy():
            return
        vs = self.vm.to_view_state()
        self._run_task("Saving review…", save_review, (self.vm.review, folder, vs), self._saved)

    def _saved(self, folder):
        self._update_title()
        self.vm.dirtyChanged.emit()
        self.page.inspector.recording.refresh()
        self.statusBar().showMessage(f"Saved review to {folder}", 10000)

    def export(self):
        rv = self.vm.review
        if rv is None:
            return
        if not rv.annotations.items:
            self.statusBar().showMessage("No user annotations to export.", 6000)
            return
        default = str(Path.home() / f"{Path(rv.emg.name).stem}_annotations.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export annotations", default, "CSV (*.csv)")
        if not path:
            return
        try:
            export_annotations(rv.annotations.items, Path(path), rv.protected_paths())
        except (OSError, PermissionError) as e:
            self._error("Export failed", str(e))
            return
        self.statusBar().showMessage(f"Exported {len(rv.annotations.items)} annotations to {path}", 8000)

    # ------------------------------------------------------------------------------------
    # navigation / selection / annotations
    # ------------------------------------------------------------------------------------
    def step(self, delta):
        if self.vm.emg is None:
            return
        self.vm.step(delta)
        self.plots.ensure_row_visible(self.vm.cursor_row)

    def center_cursor(self):
        if self.vm.cursor_row is not None:
            self.plots.center_on_x(self.vm.emg.x_of_row(self.vm.cursor_row, self.vm.axis))

    def _toggle_select(self, on):
        self.vm.select_mode = bool(on)
        self.statusBar().showMessage("Select mode ON: drag selects rows" if on else
                                     "Select mode OFF: drag pans (Shift+drag selects)", 5000)

    def zoom_selection(self):
        if self.vm.selection:
            self.plots.zoom_to_rows(*self.vm.selection)
        else:
            self.statusBar().showMessage("No selection to zoom to.", 5000)

    def goto(self):
        emg = self.vm.emg
        if emg is None:
            return
        d = GoToDialog(emg, self.vm.cursor_row, self)
        if d.exec_():
            row = d.resolve()
            if row is not None:
                self.vm.set_cursor(row)
                self.plots.ensure_row_visible(row)

    def set_start(self):
        try:
            self.vm.set_start_at_cursor()
        except SelectionError as e:
            self.statusBar().showMessage(str(e), 6000)

    def set_end(self):
        try:
            self.vm.set_end_at_cursor()
        except SelectionError as e:
            self.statusBar().showMessage(str(e), 6000)

    def store(self, which):
        if not self.vm.selection:
            self.statusBar().showMessage("Select an interval first.", 5000)
            return
        self.vm.set_ab(which, self.vm.selection)
        self.page.inspector.setCurrentWidget(self.page.inspector.compare)

    def _labels_seen(self):
        rv = self.vm.review
        return [a.label for a in rv.annotations.items if a.label] if rv else []

    def label_selection(self):
        vm = self.vm
        if vm.selection is None:
            self.statusBar().showMessage("Select an interval first (Shift+drag, or [ and ]).", 6000)
            return
        a, b = vm.selection
        d = AnnotationDialog("Label selection", f"Original rows [{a}, {b}) — {b - a:,} rows; "
                             f"inclusive last row {b - 1}; {vm.emg.run_of_row(a).name}.",
                             self._labels_seen(), parent=self)
        if d.exec_():
            v = d.values()
            self._add_annotation(a, b, TYPE_INTERVAL, v)

    def add_bookmark(self):
        vm = self.vm
        if vm.cursor_row is None:
            return
        r = vm.cursor_row
        d = AnnotationDialog("Add bookmark", f"Bookmark original row {r} "
                             f"(sample_index {vm.emg.raw_row(r)['sample_index']}).",
                             self._labels_seen(), is_bookmark=True, parent=self)
        if d.exec_():
            self._add_annotation(r, r + 1, TYPE_BOOKMARK, d.values())

    def _add_annotation(self, a, b, kind, v):
        try:
            ann = make_annotation(self.vm.emg, a, b, annotation_type=kind, label=v["label"],
                                  notes=v["notes"], channels=v["channels"],
                                  review_status=v["review_status"])
        except SelectionError as e:
            self._error("Cannot annotate", str(e))
            return
        self.vm.review.annotations.add(ann)
        self.vm.annotationsChanged.emit()
        self._update_title()

    def _find(self, key):
        try:
            return self.vm.review.annotations.get(key)
        except KeyError:
            return None

    def jump_to(self, key):
        vm = self.vm
        if key.startswith("marker:"):
            mrow = int(key.split(":")[1])
            rows = dict(vm.review.markers.emg_rows(vm.emg))[mrow]
            if not rows:
                self.statusBar().showMessage("This marker has no matching EMG row.", 6000)
                return
            vm.set_cursor(rows[0])
            self.plots.ensure_row_visible(rows[0])
            return
        a = self._find(key)
        if a is None:
            return
        vm.set_cursor(a.start_row)
        if a.annotation_type != TYPE_BOOKMARK:
            vm.set_selection(a.start_row, a.end_row_exclusive)
            self.plots.zoom_to_rows(a.start_row, a.end_row_exclusive)
        else:
            self.plots.ensure_row_visible(a.start_row)

    def edit_annotation(self, key):
        a = self._find(key) if not key.startswith("marker:") else None
        if a is None:
            self.statusBar().showMessage("Recorder markers are read-only.", 5000)
            return
        d = AnnotationDialog("Edit annotation", f"{a.annotation_type}: rows [{a.start_row}, "
                             f"{a.end_row_exclusive}), inclusive last row {a.last_row}. The row "
                             "range cannot be edited; create a new annotation instead.",
                             self._labels_seen(), annotation=a, parent=self)
        if d.exec_():
            v = d.values()
            self.vm.review.annotations.update(a.annotation_id, label=v["label"], notes=v["notes"],
                                              review_status=v["review_status"],
                                              selected_channels=channels_text(v["channels"]))
            self.vm.annotationsChanged.emit()
            self._update_title()

    def delete_annotation(self, key):
        a = self._find(key) if not key.startswith("marker:") else None
        if a is None:
            self.statusBar().showMessage("Recorder markers are read-only.", 5000)
            return
        if QMessageBox.question(self, "Delete annotation", f"Delete “{a.label or a.annotation_id}”?") \
                == QMessageBox.Yes:
            self.vm.review.annotations.delete(a.annotation_id)
            self.vm.annotationsChanged.emit()
            self._update_title()

    # ------------------------------------------------------------------------------------
    # drag and drop / close
    # ------------------------------------------------------------------------------------
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if paths:
            ev.acceptProposedAction()
            self.import_paths(paths)

    def closeEvent(self, ev):
        if self.busy():
            self.statusBar().showMessage("Please wait for the current task to finish.", 5000)
            ev.ignore()
            return
        if not self._confirm_discard():
            ev.ignore()
            return
        self.workarea.cleanup()
        ev.accept()


def run(args) -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    win = MainWindow()
    # remove the temporary work area also on Ctrl+C / SIGTERM (snapshots are copies only)
    atexit.register(win.workarea.cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.quit())
    tick = QTimer()
    tick.timeout.connect(lambda: None)        # lets Python run signal handlers
    tick.start(500)
    win.show()
    if args.review:
        QTimer.singleShot(0, lambda: win.open_review_path(args.review))
    elif args.demo:
        QTimer.singleShot(0, win.open_demo)
    elif args.paths:
        QTimer.singleShot(0, lambda: win.import_paths(args.paths))
    return app.exec_()
