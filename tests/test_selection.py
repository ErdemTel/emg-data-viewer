"""Stable row references, exact sample stepping, half-open selection and export boundaries."""

import csv
import io

import pytest

from conftest import emg_row, make_emg_text, simple_emg_rows
from emg_review.annotations import (SelectionError, annotations_from_csv, annotations_to_csv,
                                    export_annotations, make_annotation)
from emg_review.recording import AXIS_INDEX, AXIS_RECEIPT, AXIS_ROW


@pytest.fixture
def two_segments(write, load):
    rows = simple_emg_rows(50, seg=1) + simple_emg_rows(30, seg=2, start_idx=0, mono0=200.0)
    return load(write("e.csv", make_emg_text(rows)))


def test_row_numbers_are_stable_and_zero_based(two_segments):
    emg = two_segments.emg
    assert emg.raw_row(0)["sample_index"] == "0"
    assert emg.raw_row(50) == dict(zip(emg.table.columns, simple_emg_rows(30, seg=2, mono0=200.0)[0].split(",")))
    assert emg.run_of_row(49).segment == 1 and emg.run_of_row(50).segment == 2


def test_stepping_moves_by_original_rows_within_segment(two_segments):
    emg = two_segments.emg
    assert emg.step(10, +1) == 11 and emg.step(10, -1) == 9
    assert emg.step(49, +1) == 49            # does not run into segment 2
    assert emg.step(50, -1) == 50
    assert emg.step(0, -5) == 0


def test_stepping_visits_every_row_of_a_receipt_group(two_segments):
    emg = two_segments.emg
    a, b = emg.receipt_group(5)
    assert (a, b) == (0, 16)
    # all 16 rows share one receipt time, yet each is reachable by stepping
    seen, r = [a], a
    while r < b - 1:
        r = emg.step(r, 1)
        seen.append(r)
    assert seen == list(range(16))
    run = emg.runs[0]
    # clicking on the shared receipt time selects the first row of the group
    assert emg.nearest_row(run, AXIS_RECEIPT, emg.x_of_row(7, AXIS_RECEIPT)) == 0


def test_half_open_annotation_boundaries(two_segments):
    emg = two_segments.emg
    a = make_annotation(emg, 10, 20, label="pre-grasp candidate")
    assert (a.start_row, a.end_row_exclusive, a.last_row, a.n_rows) == (10, 20, 19, 10)
    assert a.start_sample_index == "10" and a.end_sample_index_inclusive == "19"
    assert a.start_host_mono == emg.raw_row(10)["host_mono"]
    assert a.end_host_mono_inclusive == emg.raw_row(19)["host_mono"]
    assert a.end_device_time_inclusive == emg.raw_row(19)["device_time"]
    assert a.segment == "1" and a.annotation_source == "user"
    assert a.source_sha256 == emg.source.sha256
    # a selection ending exactly at the segment's last row is allowed
    b = make_annotation(emg, 40, 50)
    assert b.last_row == 49


@pytest.mark.parametrize("start,stop,needle", [
    (45, 55, "cross"), (10, 10, "empty"), (20, 10, "empty"), (-1, 3, "outside"), (70, 81, "outside"),
])
def test_invalid_selections(two_segments, start, stop, needle):
    with pytest.raises(SelectionError, match=needle):
        make_annotation(two_segments.emg, start, stop)


def test_bookmark_is_one_row(two_segments):
    b = make_annotation(two_segments.emg, 7, 8, annotation_type="bookmark")
    assert b.n_rows == 1
    with pytest.raises(SelectionError):
        make_annotation(two_segments.emg, 7, 9, annotation_type="bookmark")


def test_export_roundtrip_keeps_boundaries(two_segments, tmp_path):
    emg = two_segments.emg
    items = [make_annotation(emg, 10, 20, label='label, with "quotes"', channels=[0, 2]),
             make_annotation(emg, 55, 56, annotation_type="bookmark", notes="line1\nline2")]
    path = tmp_path / "ann.csv"
    export_annotations(items, path)
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    assert rows[0]["start_row"] == "10" and rows[0]["end_row_exclusive"] == "20"
    assert rows[0]["end_sample_index_inclusive"] == "19"
    assert rows[0]["selected_channels"] == "1;3"
    back = annotations_from_csv(path.read_text())
    assert [x.__dict__ for x in back] == [x.__dict__ for x in items]
    assert annotations_to_csv(back) == path.read_text()


def test_x_range_to_rows_respects_gaps(write, load):
    rows = [emg_row(1, i if i < 10 else i + 5, 0, 0, 0) for i in range(20)]
    emg = load(write("g.csv", make_emg_text(rows))).emg
    run = emg.runs[0]
    # sample indices 10..14 are missing; x-range [8, 16] covers indices 8, 9, 15, 16
    a, b = emg.rows_in_range(run, AXIS_INDEX, 8, 16)
    assert (a, b) == (8, 12)
    assert [int(emg.sample_index[r]) for r in range(a, b)] == [8, 9, 15, 16]
    assert emg.rows_in_range(run, AXIS_ROW, 2.0, 4.0) == (2, 5)
    assert emg.nearest_row(run, AXIS_INDEX, 11.9) == 9      # nearest existing index: 9 (2.9 away)
    assert emg.nearest_row(run, AXIS_INDEX, 12.4) == 10     # 15 is 2.6 away, 9 is 3.4 away


def test_find_rows_reports_ambiguity(two_segments):
    emg = two_segments.emg
    assert emg.find_rows(5) == [5, 55]                  # index 5 exists in both segments
    assert emg.find_rows(5, segment=2) == [55]
    assert emg.find_rows(999) == []
