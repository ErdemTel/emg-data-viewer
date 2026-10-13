"""EMG plot stack, companion tracks and overview.

Drawing only: every value shown comes from the original arrays. For wide views the
curves are reduced to per-bin min/max for drawing (display only); statistics and the
sample inspector always read the original rows. Lines are never drawn across segment
runs, sample-index discontinuities or excluded (lost / masked / empty) samples.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

from ..annotations import SelectionError, TYPE_BOOKMARK
from ..diagnostics import emg_unit
from ..recording import AXIS_RECEIPT
from ..schema import N_CHANNELS
from ..stats import display_series, trailing_ac_rms

CH_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
IMU_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]
SEL_BRUSH = (90, 90, 90, 45)
A_BRUSH = (31, 119, 180, 40)
B_BRUSH = (255, 127, 14, 45)
ANN_BRUSH = (44, 160, 44, 30)
CURSOR_PEN = pg.mkPen((200, 0, 0), width=1)
ENV_PEN = pg.mkPen((20, 20, 20, 200), width=1.6)
GAP_PEN = pg.mkPen((255, 140, 0, 220), width=1, style=Qt.DashLine)
MARKER_PEN = pg.mkPen((120, 120, 120), width=1, style=Qt.DotLine)
MAX_GAP_MARKS = 200
OVERVIEW_POINTS = 3000

pg.setConfigOptions(background="w", foreground="k", antialias=False)


class EmgViewBox(pg.ViewBox):
    """Wheel: zoom X around the mouse. Drag: pan X. Shift+drag or Select mode: select rows.
    Click: inspect the nearest original row."""

    def __init__(self, area):
        super().__init__(enableMenu=False)
        self.area = area
        self.setMouseEnabled(x=True, y=False)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.LeftButton and (self.area.vm.select_mode
                                             or ev.modifiers() & Qt.ShiftModifier):
            ev.accept()
            x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
            x1 = self.mapSceneToView(ev.scenePos()).x()
            self.area.drag_select(x0, x1, ev.isFinish())
            return
        super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            ev.accept()
            self.area.click_x(self.mapSceneToView(ev.scenePos()).x())
        else:
            ev.ignore()


class OverviewViewBox(pg.ViewBox):
    def __init__(self, area):
        super().__init__(enableMenu=False)
        self.area = area
        self.setMouseEnabled(x=False, y=False)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            ev.accept()
            self.area.center_on_x(self.mapSceneToView(ev.scenePos()).x())

    def mouseDragEvent(self, ev, axis=None):
        ev.ignore()

    def wheelEvent(self, ev, axis=None):
        ev.ignore()


def _region(brush, z=-10):
    r = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(*brush), pen=pg.mkPen(None))
    r.setZValue(z)
    r.hide()
    return r


class EmgPlotArea(QWidget):
    def __init__(self, vm, parent=None):
        super().__init__(parent)
        self.vm = vm
        self._syncing = False
        self._env_cache = {}
        self._dyn = []                     # (plot, item) for annotation/marker decorations
        self._labels = []                  # (channel, TextItem, 'top'|'bottom'), re-anchored on render
        self._ylim = [(-1.0, 1.0)] * N_CHANNELS
        self._xrange = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#444; font-size: 11px;")
        lay.addWidget(self.info)
        self.split = QSplitter(Qt.Vertical)
        lay.addWidget(self.split, 1)

        # -- EMG stack ------------------------------------------------------------------
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setSpacing(0)
        self.plots, self.curves, self.env_curves, self.gap_curves = [], [], [], []
        self.cursor_lines, self.cursor_dots, self.sel_regions = [], [], []
        self.ab_regions = {"A": [], "B": []}
        for c in range(N_CHANNELS):
            p = pg.PlotItem(viewBox=EmgViewBox(self))
            p.hideButtons()
            p.setMenuEnabled(False)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.getAxis("left").setWidth(64)
            curve = pg.PlotCurveItem(pen=pg.mkPen(CH_COLORS[c], width=1))
            env = pg.PlotCurveItem(pen=ENV_PEN)
            gap = pg.PlotCurveItem(pen=GAP_PEN)
            cur = pg.InfiniteLine(angle=90, movable=False, pen=CURSOR_PEN)
            dot = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(200, 0, 0), pen=None)
            sel = _region(SEL_BRUSH, -5)
            ra, rb = _region(A_BRUSH, -8), _region(B_BRUSH, -8)
            for it in (sel, ra, rb, curve, env, gap, cur, dot):
                p.addItem(it)
            p.getViewBox().sigXRangeChanged.connect(self._on_xrange)
            self.plots.append(p)
            self.curves.append(curve)
            self.env_curves.append(env)
            self.gap_curves.append(gap)
            self.cursor_lines.append(cur)
            self.cursor_dots.append(dot)
            self.sel_regions.append(sel)
            self.ab_regions["A"].append(ra)
            self.ab_regions["B"].append(rb)
        self.split.addWidget(self.glw)

        # -- companion tracks -------------------------------------------------------------
        self.cglw = pg.GraphicsLayoutWidget()
        self.cglw.ci.setSpacing(0)
        self.imp_plot = pg.PlotItem(viewBox=EmgViewBox(self))
        self.imp_plot.setLogMode(y=True)
        self.imp_items = [pg.PlotDataItem(pen=None, symbol="o", symbolSize=5, symbolPen=None,
                                          symbolBrush=pg.mkBrush(CH_COLORS[c]))
                          for c in range(N_CHANNELS)]
        self.imp_thr = [pg.InfiniteLine(angle=0, movable=False,
                                        pen=pg.mkPen(col, width=1, style=Qt.DashLine))
                        for col in ((0, 150, 0), (230, 120, 0))]
        self.imp_cursor = pg.InfiniteLine(angle=90, movable=False, pen=CURSOR_PEN)
        for it in self.imp_items + self.imp_thr + [self.imp_cursor]:
            self.imp_plot.addItem(it)
        self.acc_plot = pg.PlotItem(viewBox=EmgViewBox(self))
        self.gyro_plot = pg.PlotItem(viewBox=EmgViewBox(self))
        self.acc_items = [self.acc_plot.plot(pen=pg.mkPen(IMU_COLORS[i]), name=n)
                          for i, n in enumerate(("acc_x", "acc_y", "acc_z"))]
        self.gyro_items = [self.gyro_plot.plot(pen=pg.mkPen(IMU_COLORS[i]), name=n)
                           for i, n in enumerate(("gyro_x", "gyro_y", "gyro_z"))]
        self.imu_cursors = []
        for p in (self.imp_plot, self.acc_plot, self.gyro_plot):
            p.hideButtons()
            p.setMenuEnabled(False)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.getAxis("left").setWidth(64)
            p.getViewBox().sigXRangeChanged.connect(self._on_xrange)
        for p in (self.acc_plot, self.gyro_plot):
            cl = pg.InfiniteLine(angle=90, movable=False, pen=CURSOR_PEN)
            p.addItem(cl)
            self.imu_cursors.append(cl)
        self.split.addWidget(self.cglw)

        # -- overview -------------------------------------------------------------------
        self.ov_vb = OverviewViewBox(self)
        self.ov = pg.PlotWidget(viewBox=self.ov_vb)
        self.ov.setMaximumHeight(120)
        self.ov.setMinimumHeight(70)
        self.ov.hideButtons()
        self.ov.getPlotItem().setMenuEnabled(False)
        self.ov.getAxis("left").setWidth(64)
        self.ov.getAxis("left").setTicks([[]])
        self.ov_curve = pg.PlotCurveItem(pen=pg.mkPen((110, 110, 110), width=1))
        self.ov_invalid = pg.ScatterPlotItem(size=6, symbol="t", brush=pg.mkBrush(220, 0, 0), pen=None)
        self.ov_gaps = pg.ScatterPlotItem(size=7, symbol="d", brush=pg.mkBrush(255, 140, 0), pen=None)
        self.ov_sel = _region(SEL_BRUSH, -5)
        self.ov_cursor = pg.InfiniteLine(angle=90, movable=False, pen=CURSOR_PEN)
        self.ov_region = pg.LinearRegionItem(brush=pg.mkBrush(0, 90, 255, 35),
                                             pen=pg.mkPen((0, 90, 255), width=1))
        self.ov_region.setZValue(10)
        self.ov_region.sigRegionChanged.connect(self._on_region)
        for it in (self.ov_sel, self.ov_curve, self.ov_invalid, self.ov_gaps, self.ov_cursor,
                   self.ov_region):
            self.ov.addItem(it)
        self.ov.setToolTip("Overview of the selected segment (display summary: max |value| across "
                           "visible channels). Drag the blue window to navigate; click to centre.")
        lay.addWidget(self.ov)
        self.split.setStretchFactor(0, 5)
        self.split.setStretchFactor(1, 2)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(15)
        self._render_timer.timeout.connect(self.render)

        vm.reviewChanged.connect(self.on_review)
        vm.runChanged.connect(self.on_run_or_axis)
        vm.axisChanged.connect(self.on_axis)
        vm.cursorChanged.connect(self.update_cursor)
        vm.selectionChanged.connect(self.update_selection)
        vm.abChanged.connect(self.update_ab)
        vm.annotationsChanged.connect(self.update_decorations)
        vm.displayChanged.connect(self.on_display)
        vm.companionsChanged.connect(self.on_companions)

    # ----------------------------------------------------------------------------------
    @property
    def all_viewboxes(self):
        return [p.getViewBox() for p in self.plots + [self.imp_plot, self.acc_plot, self.gyro_plot]]

    def _xvals(self):
        vm = self.vm
        return vm.emg.axis_values(vm.run, vm.axis)

    def _layout_plots(self):
        ci = self.glw.ci
        ci.clear()
        visible = [c for c in range(N_CHANNELS) if self.vm.channels_visible[c]]
        emg = self.vm.emg
        unit = emg_unit(self.vm.review.session_meta) if self.vm.review else None
        for i, c in enumerate(visible):
            p = self.plots[c]
            ci.addItem(p, row=i, col=0)
            p.setLabel("left", f"ch{c + 1}" + (f" [{unit}]" if unit else ""))
            last = i == len(visible) - 1
            p.showAxis("bottom", last)
            if last and emg is not None:
                p.setLabel("bottom", emg.axis_title(self.vm.axis))
        self._refit(self.glw)
        self._layout_companions()

    def _layout_companions(self):
        ci = self.cglw.ci
        ci.clear()
        r = self.vm.review
        plots = []
        if r and r.contact is not None:
            plots.append(self.imp_plot)
        if r and r.imu is not None:
            plots += [self.acc_plot, self.gyro_plot]
        heights = []
        for i, p in enumerate(plots):
            ci.addItem(p, row=i, col=0)
            p.showAxis("bottom", i == len(plots) - 1)
            h = 95 if p is self.imp_plot else 65
            ci.layout.setRowMinimumHeight(i, h)
            heights.append(h)
        self.cglw.setMinimumHeight(sum(heights) + 20)
        self.cglw.setVisible(bool(plots) and self.vm.show_companions)
        self._refit(self.cglw)

    def _default_split(self):
        h = max(400, self.split.height())
        self.split.setSizes([int(h * 0.70), int(h * 0.30)])

    @staticmethod
    def _refit(view):
        # a GraphicsView that was hidden keeps a stale scene geometry; re-apply its size
        QTimer.singleShot(0, lambda: view.resizeEvent(None))

    # ----------------------------------------------------------------------------------
    # state changes
    # ----------------------------------------------------------------------------------
    def on_review(self):
        self._env_cache.clear()
        if self.vm.review is None:
            return
        self._layout_plots()
        QTimer.singleShot(0, self._default_split)
        self.on_run_or_axis(restore=True)

    def on_axis(self):
        rows = self.vm.visible_rows
        self.on_run_or_axis(keep_rows=rows)

    def on_display(self):
        self._layout_plots()
        self._draw_overview()
        self.render()

    def on_companions(self):
        self._layout_companions()
        self._draw_companions()

    def on_run_or_axis(self, restore=False, keep_rows=None):
        vm = self.vm
        if vm.emg is None:
            return
        self._layout_plots()
        xs = self._xvals()
        fin = xs[np.isfinite(xs)]
        if len(fin) == 0:
            self.info.setText("No finite axis values in this segment for this axis mode.")
            return
        lo, hi = float(fin.min()), float(fin.max())
        d = np.diff(fin)
        d = d[d > 0]
        step = float(np.median(d)) if len(d) else 1.0
        pad = max(step, (hi - lo) * 0.01)
        for vb in self.all_viewboxes:
            vb.setLimits(xMin=lo - pad, xMax=hi + pad, minXRange=step * 4)
        self.ov.setXRange(lo - pad, hi + pad, padding=0)
        self.ov.getViewBox().setLimits(xMin=lo - pad, xMax=hi + pad)
        self.ov_region.setBounds((lo - pad, hi + pad))
        self._draw_overview()
        self._draw_companions()
        self.update_decorations()
        target = keep_rows or (vm.restore_rows if restore else None)
        vm.restore_rows = None
        run = vm.run
        if target and run.contains(target[0]) and run.contains(max(target[0], target[1] - 1)):
            self.zoom_to_rows(target[0], target[1], pad_frac=0.0)
        else:
            self.set_xrange(lo - pad, hi + pad)
        self.update_cursor()
        self.update_selection()
        self.update_ab()
        self.render()

    # ----------------------------------------------------------------------------------
    # x range sync (manual, guarded: no feedback loops between plots and overview)
    # ----------------------------------------------------------------------------------
    def set_xrange(self, x0, x1):
        self._syncing = True
        try:
            for vb in self.all_viewboxes:
                vb.setXRange(x0, x1, padding=0)
            self.ov_region.setRegion((x0, x1))
        finally:
            self._syncing = False
        self._xrange = (x0, x1)
        self._render_timer.start()

    def _on_xrange(self, vb, rng):
        if self._syncing:
            return
        self.set_xrange(*rng)

    def _on_region(self):
        if self._syncing:
            return
        x0, x1 = self.ov_region.getRegion()
        if x1 > x0:
            self.set_xrange(x0, x1)

    def current_xrange(self):
        return self.plots[0].getViewBox().viewRange()[0]

    # ----------------------------------------------------------------------------------
    # navigation helpers
    # ----------------------------------------------------------------------------------
    def fit(self):
        vm = self.vm
        if vm.emg is None:
            return
        xs = self._xvals()
        fin = xs[np.isfinite(xs)]
        if len(fin):
            pad = max(1e-9, (fin.max() - fin.min()) * 0.01)
            self.set_xrange(fin.min() - pad, fin.max() + pad)

    def zoom_to_rows(self, a, b, pad_frac=0.05):
        vm = self.vm
        emg = vm.emg
        xa, xb = emg.x_of_row(a, vm.axis), emg.x_of_row(max(a, b - 1), vm.axis)
        if not (np.isfinite(xa) and np.isfinite(xb)):
            vm.message.emit("The selection has no finite coordinates on this axis.")
            return
        if xb <= xa:          # one row or identical receipt times: show ±50 rows around it
            run = emg.run_of_row(a)
            lo, hi = max(run.start, a - 50), min(run.stop - 1, b + 50)
            xa, xb = emg.x_of_row(lo, vm.axis), emg.x_of_row(hi, vm.axis)
            if xb <= xa:
                xb = xa + 1
        pad = (xb - xa) * pad_frac
        self.set_xrange(xa - pad, xb + pad)

    def center_on_x(self, x):
        x0, x1 = self.current_xrange()
        w = x1 - x0
        self.set_xrange(x - w / 2, x + w / 2)

    def ensure_row_visible(self, row):
        vm = self.vm
        x = vm.emg.x_of_row(row, vm.axis)
        if not np.isfinite(x):
            return
        x0, x1 = self.current_xrange()
        if not (x0 <= x <= x1):
            self.center_on_x(x)

    def pan_fraction(self, frac):
        x0, x1 = self.current_xrange()
        w = x1 - x0
        self.set_xrange(x0 + frac * w, x1 + frac * w)

    def zoom_factor(self, factor):
        x0, x1 = self.current_xrange()
        vm = self.vm
        c = (x0 + x1) / 2
        if vm.cursor_row is not None:
            xc = vm.emg.x_of_row(vm.cursor_row, vm.axis)
            if np.isfinite(xc) and x0 <= xc <= x1:
                c = xc
        self.set_xrange(c - (c - x0) * factor, c + (x1 - c) * factor)

    # ----------------------------------------------------------------------------------
    # mouse interactions
    # ----------------------------------------------------------------------------------
    def click_x(self, x):
        vm = self.vm
        if vm.emg is None:
            return
        row = vm.emg.nearest_row(vm.run, vm.axis, x)
        if row is not None:
            vm.set_cursor(row)

    def drag_select(self, x0, x1, finished):
        vm = self.vm
        if vm.emg is None:
            return
        a, b = vm.emg.rows_in_range(vm.run, vm.axis, min(x0, x1), max(x0, x1))
        if b <= a:
            return
        try:
            vm.pending_start = None
            vm.set_selection(a, b)
        except SelectionError as e:
            vm.message.emit(str(e))

    # ----------------------------------------------------------------------------------
    # drawing
    # ----------------------------------------------------------------------------------
    def _envelope(self):
        vm = self.vm
        key = (vm.run_index, vm.envelope_window)
        if key not in self._env_cache:
            self._env_cache.clear()
            self._env_cache[key] = trailing_ac_rms(vm.emg, vm.run, vm.envelope_window)
        return self._env_cache[key]

    def render(self):
        vm = self.vm
        emg = vm.emg
        if emg is None:
            return
        run = vm.run
        xs_all = self._xvals()
        x0, x1 = self.current_xrange()
        a, b = emg.rows_in_range(run, vm.axis, x0, x1)
        vm.visible_rows = (a, b) if b > a else None
        a2, b2 = max(run.start, a - 1), min(run.stop, b + 1)
        if not emg.axis_monotonic(run, vm.axis):
            a2, b2 = run.start, run.stop
        sl = slice(a2 - run.start, b2 - run.start)
        x = xs_all[sl]
        link = emg.contiguous_after[a2:b2]
        width = max(200, int(self.glw.width()))
        max_points = 2 * width
        env = self._envelope()[sl] if vm.envelope_on else None
        visible = [c for c in range(N_CHANNELS) if vm.channels_visible[c]]
        reduced = False
        lims = {}
        for c in range(N_CHANNELS):
            if c not in visible:
                continue
            y = np.where(emg.eligible[a2:b2, c], emg.X[a2:b2, c], np.nan)
            xs, ys, conn = display_series(x, y, link, max_points)
            reduced |= (b2 - a2) > max_points
            self.curves[c].setData(xs, ys, connect=conn, skipFiniteCheck=True)
            yv = y[(a - a2):(b - a2)] if b > a else y
            fin = yv[np.isfinite(yv)]
            lo, hi = (float(fin.min()), float(fin.max())) if len(fin) else (np.nan, np.nan)
            if env is not None:
                ex, ey, ec = display_series(x, env[:, c], link, max_points)
                self.env_curves[c].setData(ex, ey, connect=ec, skipFiniteCheck=True)
                self.env_curves[c].show()
            else:
                self.env_curves[c].hide()
            lims[c] = (lo, hi)
        self._apply_ylims(lims)
        self._draw_gaps(a2, b2, x, run)
        self._place_labels()
        n_vis = max(0, b - a)
        txt = (f"{run.name}: showing original rows {a}–{b - 1} ({n_vis:,} rows)" if n_vis
               else f"{run.name}: no rows in view")
        if reduced:
            txt += " · wide view drawn as per-pixel min/max (display only; inspection and statistics use every original sample)"
        if vm.envelope_on:
            txt += f" · black = derived envelope, trailing AC RMS over {vm.envelope_window} samples (display only)"
        if vm.y_mode == "equal":
            txt += " · equal vertical scale"
        else:
            txt += " · INDEPENDENT vertical scales — amplitudes are not comparable between channels"
        if self.vm.review and emg_unit(self.vm.review.session_meta) is None:
            txt += " · EMG units: not recorded in the files"
        self.info.setText(txt)

    def _apply_ylims(self, lims):
        vm = self.vm
        if not lims:
            return
        if vm.y_mode == "equal":
            los = [v[0] for v in lims.values() if np.isfinite(v[0])]
            his = [v[1] for v in lims.values() if np.isfinite(v[1])]
            lo, hi = (min(los), max(his)) if los else (-1.0, 1.0)
            lims = {c: (lo, hi) for c in lims}
        for c, (lo, hi) in lims.items():
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo, hi = -1.0, 1.0
            if hi <= lo:
                lo, hi = lo - 1.0, hi + 1.0
            pad = (hi - lo) * 0.05
            self._ylim[c] = (lo - pad, hi + pad)
            self.plots[c].getViewBox().setYRange(lo - pad, hi + pad, padding=0)

    def _draw_gaps(self, a2, b2, x, run):
        emg = self.vm.emg
        idx = np.flatnonzero(~emg.contiguous_after[a2:b2 - 1]) if b2 - a2 > 1 else np.empty(0, int)
        idx = idx[(a2 + idx) < run.stop - 1]
        if len(idx) == 0 or len(idx) > MAX_GAP_MARKS:
            for g in self.gap_curves:
                g.setData([], [])
            return
        gx = 0.5 * (x[idx] + x[idx + 1])
        gx = gx[np.isfinite(gx)]
        for c, g in enumerate(self.gap_curves):
            lo, hi = self._ylim[c]
            g.setData(np.repeat(gx, 2), np.tile([lo, hi], len(gx)),
                      connect=np.tile([1, 0], len(gx)).astype(np.int32))

    def _place_labels(self):
        for c, t, where in self._labels:
            lo, hi = self._ylim[c]
            t.setPos(t.pos().x(), hi if where == "top" else lo)

    def _draw_overview(self):
        vm = self.vm
        emg = vm.emg
        run = vm.run
        if emg is None:
            return
        x = self._xvals()
        vis = [c for c in range(N_CHANNELS) if vm.channels_visible[c]] or list(range(N_CHANNELS))
        Y = np.where(emg.eligible[run.start:run.stop][:, vis], np.abs(emg.X[run.start:run.stop][:, vis]), np.nan)
        with np.errstate(all="ignore"):
            m = np.nanmax(np.where(np.isnan(Y), -np.inf, Y), axis=1)
        m[~np.isfinite(m)] = np.nan
        xs, ys, conn = display_series(x, m, emg.contiguous_after[run.start:run.stop], OVERVIEW_POINTS)
        self.ov_curve.setData(xs, ys, connect=conn, skipFiniteCheck=True)
        top = float(np.nanmax(m)) if np.isfinite(m).any() else 1.0
        self.ov.setYRange(-0.08 * top, top * 1.05, padding=0)
        bad = np.flatnonzero(emg.lost_flag[run.start:run.stop]
                             | emg.mask_excluded[run.start:run.stop].any(axis=1))
        if len(bad) > 5000:
            bad = bad[np.linspace(0, len(bad) - 1, 5000).astype(int)]
        self.ov_invalid.setData(x[bad], np.full(len(bad), -0.04 * top))
        gaps = np.flatnonzero(~emg.contiguous_after[run.start:run.stop - 1])
        self.ov_gaps.setData(0.5 * (x[gaps] + x[gaps + 1]) if len(gaps) else [],
                             np.full(len(gaps), -0.04 * top))

    def _rows_x(self, a, last):
        vm = self.vm
        return vm.emg.x_of_row(a, vm.axis), vm.emg.x_of_row(last, vm.axis)

    def update_cursor(self):
        vm = self.vm
        if vm.emg is None or vm.cursor_row is None or not vm.run.contains(vm.cursor_row):
            for it in self.cursor_lines + self.cursor_dots + [self.ov_cursor, self.imp_cursor] + self.imu_cursors:
                it.hide()
            return
        r = vm.cursor_row
        x = vm.emg.x_of_row(r, vm.axis)
        ok = bool(np.isfinite(x))
        for c in range(N_CHANNELS):
            self.cursor_lines[c].setVisible(ok)
            if ok:
                self.cursor_lines[c].setPos(x)
            if ok and vm.emg.eligible[r, c]:
                self.cursor_dots[c].setData([x], [vm.emg.X[r, c]])
                self.cursor_dots[c].show()
            else:
                self.cursor_dots[c].hide()
        for it in [self.ov_cursor, self.imp_cursor] + self.imu_cursors:
            it.setVisible(ok)
            if ok:
                it.setPos(x)

    def _set_region(self, items, rows):
        if rows is None:
            for it in items:
                it.hide()
            return
        xa, xb = self._rows_x(rows[0], rows[1] - 1)
        if not (np.isfinite(xa) and np.isfinite(xb)) or not self.vm.run.contains(rows[0]):
            for it in items:
                it.hide()
            return
        for it in items:
            it.setRegion((xa, xb))
            it.show()

    def update_selection(self):
        if self.vm.emg is None:
            return
        self._set_region(self.sel_regions + [self.ov_sel], self.vm.selection)

    def update_ab(self):
        if self.vm.emg is None:
            return
        for k in ("A", "B"):
            self._set_region(self.ab_regions[k], self.vm.ab[k])

    @staticmethod
    def _small_font():
        f = QFont()
        f.setPointSize(8)
        return f

    def _bottom_plot(self):
        for c in reversed(range(N_CHANNELS)):
            if self.vm.channels_visible[c]:
                return self.plots[c]
        return self.plots[-1]

    def _top_plot(self):
        for c in range(N_CHANNELS):
            if self.vm.channels_visible[c]:
                return self.plots[c]
        return self.plots[0]

    def update_decorations(self):
        """User annotations (bands / bookmark lines) and recorder markers."""
        for p, it in self._dyn:
            p.removeItem(it)
        self._dyn = []
        self._labels = []
        vm = self.vm
        r = vm.review
        if r is None:
            return
        emg, run = r.emg, vm.run
        top = self._top_plot()
        bottom = self._bottom_plot()
        for a in r.annotations.sorted():
            if not run.contains(a.start_row) or a.source_sha256 != emg.source.sha256:
                continue
            xa, xb = self._rows_x(a.start_row, a.last_row)
            if not (np.isfinite(xa) and np.isfinite(xb)):
                continue
            for p in self.plots + [self.ov.getPlotItem()]:
                if a.annotation_type == TYPE_BOOKMARK:
                    it = pg.InfiniteLine(pos=xa, angle=90, movable=False,
                                         pen=pg.mkPen((44, 130, 44), width=1.5, style=Qt.DashLine))
                else:
                    it = pg.LinearRegionItem(values=(xa, xb), movable=False,
                                             brush=pg.mkBrush(*ANN_BRUSH),
                                             pen=pg.mkPen((44, 160, 44, 120)))
                    it.setZValue(-20)
                p.addItem(it)
                self._dyn.append((p, it))
            t = pg.TextItem(a.label or "(unlabelled)", color=(30, 110, 30), anchor=(0, 0))
            t.setFont(self._small_font())
            t.setPos(xa, self._ylim[self.plots.index(top)][1])
            top.addItem(t)
            self._dyn.append((top, t))
            self._labels.append((self.plots.index(top), t, "top"))
        if r.markers is not None:
            for mrow, rows in r.markers.emg_rows(emg):
                rows = [x for x in rows if run.contains(x)]
                if not rows:
                    continue
                x = emg.x_of_row(rows[0], vm.axis)
                if not np.isfinite(x):
                    continue
                for p in self.plots:
                    it = pg.InfiniteLine(pos=x, angle=90, movable=False, pen=MARKER_PEN)
                    p.addItem(it)
                    self._dyn.append((p, it))
                t = pg.TextItem("marker: " + r.markers.labels[mrow], color=(90, 90, 90), anchor=(0, 1))
                t.setFont(self._small_font())
                t.setPos(x, self._ylim[self.plots.index(bottom)][0])
                t.setToolTip("Recorder marker (read-only). Its emg_sample_index is the LAST RECEIVED "
                             "EMG sample when it was logged — not an acquisition-time event.")
                bottom.addItem(t)
                self._dyn.append((bottom, t))
                self._labels.append((self.plots.index(bottom), t, "bottom"))

    def _track_x(self, track):
        """x coordinates for a companion track in the current run and axis mode."""
        vm = self.vm
        emg, run = vm.emg, vm.run
        if vm.axis == AXIS_RECEIPT:
            res = track.receipt_x(emg, run)
            if res is not None:
                return res[0], res[1], "plotted at their own receipt time"
        obs, rows, unplaced = track.placement(emg, run)
        x = self._xvals()[rows - run.start] if len(rows) else np.empty(0)
        return obs, x, ("placed at the first EMG row received at/after each observation "
                        "(receipt-based association, not acquisition-aligned)")

    def _draw_companions(self):
        vm = self.vm
        r = vm.review
        if r is None:
            return
        self._layout_companions()
        if r.contact is not None:
            ct = r.contact
            why = ct.association_unavailable(r.emg)
            obs, x, how = self._track_x(ct) if not why else (np.empty(0, int), np.empty(0), why)
            k = ct.kohm()
            use = ct.usable()
            n_unknown = 0
            for c in range(N_CHANNELS):
                ok = use[obs, c] if len(obs) else np.empty(0, bool)
                n_unknown += int((~ok).sum())
                self.imp_items[c].setData(x[ok] if len(obs) else [], k[obs[ok], c] if len(obs) else [])
            lo, hi = vm.contact_policy_kohm
            self.imp_thr[0].setPos(np.log10(lo))
            self.imp_thr[1].setPos(np.log10(hi))
            vals = k[obs][use[obs]] if len(obs) else np.empty(0)
            ymin = min([lo] + ([float(vals.min())] if len(vals) else []))
            ymax = max([hi] + ([float(vals.max())] if len(vals) else []))
            self.imp_plot.getViewBox().setYRange(np.log10(ymin) - 0.15, np.log10(ymax) + 0.15, padding=0)
            nice = [v * 10 ** e for e in range(-1, 7) for v in (1, 2, 5)]
            lo_t, hi_t = ymin / 1.5, ymax * 1.5
            ticks = [(np.log10(v), f"{v:g}") for v in nice if lo_t <= v <= hi_t]
            self.imp_plot.getAxis("left").setTicks([ticks, []])
            self.imp_plot.setLabel("left", "kΩ (log)")
            self.imp_plot.setTitle(
                f"<span style='font-size:8pt'>ELECTRODE–SKIN IMPEDANCE, logged observations "
                f"(Ω recorded, kΩ shown) · dashed = display policy {lo:g}/{hi:g} kΩ · "
                f"unknown not plotted: {n_unknown}</span>")
            self.imp_plot.setToolTip(
                f"Each dot is one LOGGED observation from contact.csv ({how}). Observations are "
                "matched only within the same segment. The recorder logs cached values when they "
                "change, so an unchanged value does not show a new measurement. Colour thresholds "
                "are a configurable display policy, not a signal-validity rule.")
        if r.imu is not None:
            im = r.imu
            why = im.association_unavailable(r.emg)
            obs, x, how = self._track_x(im) if not why else (np.empty(0, int), np.empty(0), why)
            for i, it in enumerate(self.acc_items):
                it.setData(x, im.A[obs, i] if len(obs) else [], connect="finite")
            for i, it in enumerate(self.gyro_items):
                it.setData(x, im.A[obs, 3 + i] if len(obs) else [], connect="finite")
            self.acc_plot.setLabel("left", "acc")
            self.gyro_plot.setLabel("left", "gyro")
            self.acc_plot.setTitle("<span style='font-size:8pt'>IMU (red/green/blue = x/y/z, values "
                                   "as recorded; units not stated in the files)</span>")
            for p in (self.acc_plot, self.gyro_plot):
                p.setToolTip(f"IMU rows {how}.")
