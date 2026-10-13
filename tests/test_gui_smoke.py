"""Offscreen GUI smoke test: drives the real window with the synthetic demo.

Runs with QT_QPA_PLATFORM=offscreen (set below if unset). This exercises the widgets and
their wiring; it does not replace a manual check of mouse drag-and-drop on a real desktop.
"""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

from conftest import CONTACT_HEADER, make_emg_text, simple_emg_rows  # noqa: E402
from emg_review.recording import AXIS_INDEX, AXIS_NOMINAL, AXIS_RECEIPT, AXIS_ROW  # noqa: E402
from emg_review.snapshot import WorkArea  # noqa: E402
from emg_review.stats import channel_stats  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    from emg_review.gui.main_window import MainWindow
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    w = MainWindow(WorkArea(tmp_path / "work"))
    w.errors = []
    w._error = lambda *a, **k: w.errors.append(a)
    w.resize(1600, 1000)
    w.show()
    QTest.qWait(50)
    yield w
    if w.vm.review:
        w.vm.review.mark_clean()
    w.close()
    QTest.qWait(10)


def wait(w, timeout=30):
    t0 = time.time()
    while w.busy() and time.time() - t0 < timeout:
        QTest.qWait(20)
    QTest.qWait(150)
    assert not w.busy()


def test_launch_is_empty(win):
    assert win.vm.review is None
    assert win.stack.currentWidget() is win.welcome
    assert not win.a["save"].isEnabled()


def test_demo_interactions_save_and_reopen(win, app, tmp_path):
    from emg_review.gui.main_window import MainWindow
    t0 = time.perf_counter()
    win.open_demo()
    wait(win)
    load_s = time.perf_counter() - t0
    vm, area = win.vm, win.plots
    emg = vm.emg
    assert emg.n == 70_032 and "SYNTHETIC" in win.windowTitle()
    assert win.stack.currentWidget() is win.page
    assert len(vm.review.annotations.items) == 3 and not vm.review.dirty

    # wide view: drawn reduced, never more than ~2 points per pixel per channel
    t1 = time.perf_counter()
    area.render()
    render_full_s = time.perf_counter() - t1
    n_pts = len(area.curves[0].xData)
    assert 0 < n_pts <= 2 * max(200, area.glw.width()) + 200

    # click -> nearest original row; arrows step one ORIGINAL row (also inside receipt groups)
    x = emg.x_of_row(1000, vm.axis)
    area.click_x(x)
    assert vm.cursor_row == 1000
    win.step(1)
    assert vm.cursor_row == 1001
    win.step(-10)
    assert vm.cursor_row == 991
    a, b = emg.receipt_group(991)
    assert b - a > 1
    t = win.page.inspector.sample.table
    fields = [t.item(i, 0).text() for i in range(t.rowCount())]
    assert "Original data row" in fields and "imp1" in fields and "Receipt group" in fields

    # zoomed-in view draws the original samples (no reduction)
    area.zoom_to_rows(2000, 2400, pad_frac=0)
    area.render()
    assert len(area.curves[0].xData) <= 402
    xs = area.curves[0].xData
    assert set(np.round(xs).astype(int)) <= set(emg.sample_index[1999:2401].tolist())

    # drag-select and statistics equal the core computation
    area.drag_select(emg.x_of_row(2100, vm.axis), emg.x_of_row(2299, vm.axis), True)
    assert vm.selection == (2100, 2300)
    QTest.qWait(150)
    st = channel_stats(emg, 2100, 2300)
    sel_table = win.page.inspector.selection.table
    assert sel_table.rowCount() == 8
    assert sel_table.item(0, 3).text() == f"{st[0].ac_rms:.4g}"

    # set start/end at the cursor: end is the INCLUSIVE last row
    vm.set_cursor(2050)
    win.set_start()
    vm.set_cursor(2060)
    win.set_end()
    assert vm.selection == (2050, 2061)

    # A/B comparison
    win.store("A")
    area.drag_select(emg.x_of_row(5000, vm.axis), emg.x_of_row(5199, vm.axis), True)
    win.store("B")
    cmp = win.page.inspector.compare
    assert cmp.table.rowCount() == 8

    # user annotation + bookmark
    win._add_annotation(5000, 5200, "interval", {"label": "pre-grasp candidate", "notes": "n",
                                                 "channels": [0, 1], "review_status": "candidate"})
    win._add_annotation(5100, 5101, "bookmark", {"label": "", "notes": "", "channels": list(range(8)),
                                                 "review_status": "unreviewed"})
    assert vm.review.dirty and "*" in win.windowTitle()
    assert win.page.inspector.annotations.table.rowCount() == 5 + 3     # 5 user + 3 markers

    # axis modes, channel visibility, envelope, independent scale
    for mode in (AXIS_ROW, AXIS_RECEIPT, AXIS_NOMINAL, AXIS_INDEX):
        vm.set_axis(mode)
        area.render()
        assert area.cursor_lines[0].value() == pytest.approx(emg.x_of_row(vm.cursor_row, mode))
    vm.set_axis(AXIS_RECEIPT)
    vm.set_channel_visible(2, False)
    assert len(area.glw.ci.items) == 7
    vm.set_envelope(True, 64)
    area.render()
    assert len(area.env_curves[0].xData) > 0
    vm.set_y_mode("independent")
    vm.set_run(1)
    assert vm.run.segment == 2 and vm.cursor_row == vm.run.start

    # save, then reopen in a fresh window: annotations and view settings restored
    folder = tmp_path / "saved_review"
    win._save_to(folder)
    wait(win)
    assert (folder / "manifest.json").exists() and not vm.review.dirty
    saved_state = vm.to_view_state()
    w2 = MainWindow(WorkArea(tmp_path / "work2"))
    w2._error = lambda *a, **k: None
    w2.show()
    w2.open_review_path(folder)
    wait(w2)
    vm2 = w2.vm
    assert len(vm2.review.annotations.items) == 5
    assert vm2.axis == AXIS_RECEIPT and vm2.run_index == 1
    assert vm2.channels_visible == saved_state["channels_visible"]
    assert vm2.envelope_on and vm2.envelope_window == 64 and vm2.y_mode == "independent"
    assert vm2.ab["A"] == (2050, 2061) and vm2.ab["B"] == (5000, 5200)
    w2.close()
    print(f"\n[offscreen GUI, demo 70,032 rows] generate+import+first draw {load_s:.2f} s; "
          f"full-segment render() {render_full_s * 1000:.0f} ms; {n_pts} drawn points/channel")


def test_drop_emg_then_add_companion_and_reject_second_emg(win, tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "emg.csv").write_text(make_emg_text(simple_emg_rows(200)))
    win.import_paths([str(rec / "emg.csv")])
    wait(win)
    assert win.vm.emg.n == 200 and win.vm.review.contact is None
    (rec / "contact.csv").write_text(CONTACT_HEADER + "\n1700000000.0,1,1e5,1e5,1e5,1e5,1e5,1e5,1e5,1e5\n")
    win.import_paths([str(rec / "contact.csv")])
    wait(win)
    assert win.vm.review.contact is not None
    assert win.vm.review.dirty
    # an unrelated file is reported, not silently ignored
    (rec / "notes.csv").write_text("a,b\n1,2\n")
    win.import_paths([str(rec / "notes.csv")])
    wait(win)
    assert win.errors and "Nothing imported" in win.errors[-1][0]


def test_companions_without_emg_are_refused(win, tmp_path):
    p = tmp_path / "contact.csv"
    p.write_text(CONTACT_HEADER + "\n1700000000.0,1,1,1,1,1,1,1,1,1\n")
    win.import_paths([str(p)])
    wait(win)
    assert win.vm.review is None
    assert win.errors and "No EMG" in win.errors[-1][0]
    assert not list((win.workarea.root / "sources").iterdir())      # snapshots discarded
