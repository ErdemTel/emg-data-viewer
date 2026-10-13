"""Electrode-skin impedance / markers / IMU association: segment-aware, receipt-based, honest."""

import numpy as np

from conftest import CONTACT_HEADER, make_emg_text, simple_emg_rows
from emg_review.diagnostics import contact_summary
from emg_review.workspace import import_issues, import_paths


def _setup(write, load, contact_lines, markers=None):
    rows = simple_emg_rows(64, seg=1, mono0=100.0, wall0=1_700_000_000.0) + \
        simple_emg_rows(64, seg=2, mono0=200.0, wall0=1_700_000_100.0)
    paths = [write("r/emg.csv", make_emg_text(rows)),
             write("r/contact.csv", CONTACT_HEADER + "\n" + "\n".join(contact_lines) + "\n")]
    if markers:
        paths.append(write("r/markers.csv", "host_time,host_mono,segment,emg_sample_index,label\n"
                           + "\n".join(markers) + "\n"))
    return load(*paths)


def test_asof_never_crosses_segments(write, load):
    # seg 1 has an observation; seg 2's first observation comes 60 ms after its start
    r = _setup(write, load, [
        "1700000000.000000,1,100000,1,1,1,1,1,1,1",
        "1700000100.060000,2,900000,1,1,1,1,1,1,1",
    ])
    emg, ct = r.emg, r.contact
    m, why = ct.asof(emg, 70)          # seg 2, receipt 200.00 s: before seg 2's observation
    assert m is None and "segment 2" in why
    m, _ = ct.asof(emg, 64 + 40)       # seg 2 row received at 200.06
    assert m.obs_row == 1
    m, _ = ct.asof(emg, 63)            # last row of segment 1
    assert m.obs_row == 0
    assert ct.Z[1, 0] == 900000.0      # recorded Ohm unchanged; kOhm only for display
    assert ct.kohm()[1, 0] == 900.0


def test_age_is_since_logged_observation_on_receipt_clock(write, load):
    r = _setup(write, load, ["1700000000.030000,1,1,2,3,4,5,6,7,8"])
    emg, ct = r.emg, r.contact
    m, _ = ct.asof(emg, 40)            # row 40 received at 100.06 (group 2)
    assert m.clock == "host_time"      # shared clock with contact.csv (v1 has host_time only)
    assert abs(m.age_s - 0.030) < 1e-6
    m, why = ct.asof(emg, 5)           # received before the observation was logged
    assert m is None and "at or before" in why


def test_unknown_nonpositive_impedance_not_usable(write, load):
    r = _setup(write, load, ["1700000000.0,1,,0,-5,nan,inf,100,200,300"])
    usable = r.contact.usable()[0]
    assert list(usable) == [False, False, False, False, False, True, True, True]
    s = contact_summary(r.contact)
    assert s.unknown[:5] == [1, 1, 1, 1, 1]
    assert "does not show that a new measurement was made" in s.note


def test_placement_uses_first_row_received_at_or_after(write, load):
    r = _setup(write, load, ["1700000000.030000,1,1,1,1,1,1,1,1,1",
                             "1700000000.500000,1,2,2,2,2,2,2,2,2",
                             "1700000100.000000,2,3,3,3,3,3,3,3,3"])
    emg, ct = r.emg, r.contact
    obs, rows, unplaced = ct.placement(emg, emg.runs[0])
    assert list(obs) == [0] and list(rows) == [16]   # second obs is after the last seg-1 row
    assert unplaced == 1
    obs2, rows2, _ = ct.placement(emg, emg.runs[1])
    assert list(obs2) == [2] and list(rows2) == [64]


def test_contact_without_segment_has_no_association(write, load):
    rows = simple_emg_rows(32)
    r = load(write("n/emg.csv", make_emg_text(rows)),
             write("n/contact.csv", "host_time,imp1,imp2,imp3,imp4,imp5,imp6,imp7,imp8\n"
                   "1700000000.0,1,1,1,1,1,1,1,1\n"))
    m, why = r.contact.asof(r.emg, 10)
    assert m is None and "no segment column" in why


def test_markers_keep_their_timing_semantics(write, load):
    r = _setup(write, load, ["1700000000.0,1,1,1,1,1,1,1,1,1"],
               markers=["1700000000.045,100.045,1,31,cue", "1700000100.01,200.01,2,5,cue2",
                        "1700000000.05,100.05,1,999,unmatched"])
    mk = r.markers
    pairs = mk.emg_rows(r.emg)
    assert pairs[0][1] == [31] and pairs[1][1] == [64 + 5] and pairs[2][1] == []


def test_inconsistent_companions_are_flagged(write, workarea):
    rows = simple_emg_rows(32, seg=1)
    e = write("x/emg.csv", make_emg_text(rows))
    c = write("x/contact.csv", CONTACT_HEADER + "\n1800000000.0,7,1,1,1,1,1,1,1,1\n")
    s = write("x/session.json", '{"session_id": "abc", "counts": {"emg": {"written": 999}}}')
    out = import_paths([e, c, s], workarea)
    issues = import_issues(out)
    text = " | ".join(issues)
    assert "segment(s) [7]" in text and "does not overlap" in text and "999" in text


def test_imu_track_placement(write, load):
    rows = simple_emg_rows(64, seg=1, mono0=100.0)
    imu = ("segment,sample_index,device_time,host_time,host_mono,valid_mask,"
           "acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n"
           "1,0,1.0,1700000000.005,100.005,63,0,0,1,0,0,0\n"
           "1,1,1.005,1700000000.035,100.035,63,0,0,1,0,0,0\n")
    r = load(write("i/emg.csv", make_emg_text(rows)), write("i/imu.csv", imu))
    obs, erow, _ = r.imu.placement(r.emg, r.emg.runs[0])
    assert list(erow) == [16, 32]                # first EMG rows received at/after each IMU row
    rows_x = r.imu.receipt_x(r.emg, r.emg.runs[0])
    assert np.allclose(rows_x[1], [0.005, 0.035])
