"""Parsing the recorder schema: optional files/fields, malformed input, duplicates, gaps, segments."""

import numpy as np
import pytest

from conftest import CONTACT_HEADER, EMG_HEADER, emg_row, make_emg_text, simple_emg_rows
from emg_review.csvtable import CsvFormatError, parse_csv
from emg_review.recording import AXIS_INDEX, AXIS_ROW
from emg_review.schema import KIND_CONTACT, KIND_EMG, KIND_MARKERS, KIND_SESSION
from emg_review.workspace import import_paths


def test_current_schema_parses_exactly(write, load):
    rows = simple_emg_rows(40)
    p = write("rec/emg.csv", make_emg_text(rows))
    r = load(p)
    emg = r.emg
    assert emg.n == 40
    assert emg.sample_index.dtype == np.int64
    assert list(emg.sample_index) == list(range(40))
    # exact original text is preserved for inspection
    raw = emg.raw_row(17)
    assert raw["host_mono"] == rows[17].split(",")[4]
    assert raw["device_time"] == rows[17].split(",")[2]
    assert emg.X[17, 0] == float(rows[17].split(",")[7])
    assert emg.has_lost and emg.has_mask
    assert emg.eligible.all()


def test_recognition_by_columns_not_filename(write, workarea):
    emg = write("x/recording_data.txt", make_emg_text(simple_emg_rows(10)))
    contact = write("x/emg.csv", CONTACT_HEADER + "\n1700000000.0,1,1,2,3,4,5,6,7,8\n")
    out = import_paths([emg, contact], workarea)
    kinds = {s.original_name: s.kind for s in out.sources}
    assert kinds == {"recording_data.txt": KIND_EMG, "emg.csv": KIND_CONTACT}


def test_missing_optional_fields_and_companions(write, load):
    # only the required columns: no segment, timing, lost or valid_mask
    text = "sample_index,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n" + "\n".join(
        f"{i},{i},{i},{i},{i},{i},{i},{i},{i}" for i in range(5)) + "\n"
    r = load(write("min.csv", text))
    emg = r.emg
    assert not emg.has_segment and emg.runs[0].segment is None
    assert emg.available_axes() == [AXIS_INDEX, AXIS_ROW]      # no time axes invented
    assert emg.receipt_column() is None
    assert emg.validity_unknown.all()                             # unknown, not "valid"
    assert "unknown" in emg.validity(0, 0)
    assert r.contact is None and r.imu is None and r.markers is None and r.session_meta is None


def test_extra_columns_are_accepted_and_preserved(write, load):
    header = EMG_HEADER + ",callback_id,block_id,my_note"
    rows = [emg_row(1, i, 1.0 + i / 1000, 2.0, 3.0) + f",{i // 4},{i // 4},n{i}" for i in range(8)]
    r = load(write("v2.csv", header + "\n" + "\n".join(rows) + "\n"))
    assert r.emg.extra_columns == ["callback_id", "block_id", "my_note"]
    assert r.emg.raw_row(5)["my_note"] == "n5"
    assert r.emg.callback_id is not None


def test_insufficient_emg_columns_is_a_clear_error(write, workarea):
    p = write("bad.csv", "segment,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n1,1,2,3,4,5,6,7,8\n")
    out = import_paths([p], workarea)
    assert out.errors and "sample_index" in out.errors[0]
    assert not out.sources


@pytest.mark.parametrize("text,needle", [
    (EMG_HEADER + "\n" + emg_row(1, 0, 1, 2, 3) + "\n1,1,2\n", "has 3 fields"),
    (EMG_HEADER + "\n" + emg_row(1, 0, 1, 2, 3).replace("1,0,", "1,abc,", 1) + "\n", "integer"),
    (EMG_HEADER + "\n" + emg_row(1, 0, 1, 2, 3).replace("1,0,", ",0,", 1) + "\n", "segment"),
    ("", "empty"),
])
def test_malformed_emg_is_rejected_not_dropped(write, workarea, text, needle):
    p = write("m.csv", text)
    out = import_paths([p], workarea)
    assert not out.sources
    msg = " ".join(out.errors + [r for _, r in out.rejected])
    assert needle in msg


def test_unreadable_and_binary_files(write, workarea, tmp_path):
    b = write("blob.csv", b"\x00\x01\x02binary")
    nonutf = write("latin.csv", ("sample_index,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n0,1,2,3,4,5,6,7,8\n"
                                 "1,\xe9,2,3,4,5,6,7,8\n").encode("latin-1"))
    out = import_paths([b, nonutf, tmp_path / "does_not_exist.csv"], workarea)
    reasons = dict(out.rejected)
    assert "binary" in reasons["blob.csv"]
    assert "UTF-8" in reasons["latin.csv"]
    assert any("not found" in n for n in out.notes)


def test_non_numeric_values_kept_visible_not_dropped(write, load):
    rows = simple_emg_rows(6)
    rows[3] = rows[3].rsplit(",", 1)[0] + ",oops"
    rows[4] = rows[4].rsplit(",", 1)[0] + ",nan"
    r = load(write("e.csv", make_emg_text(rows)))
    emg = r.emg
    assert emg.n == 6                          # no row dropped
    assert np.isnan(emg.X[3, 7]) and emg.raw_row(3)["ch8"] == "oops"
    assert emg.table.bad_cells["ch8"].count == 1
    assert not emg.eligible[3, 7] and not emg.eligible[4, 7]


def test_blank_lines_are_reported_and_rows_keep_numbers(write, load):
    rows = simple_emg_rows(4)
    text = EMG_HEADER + "\n" + rows[0] + "\n" + rows[1] + "\n\n" + rows[2] + "\n" + rows[3] + "\n"
    r = load(write("b.csv", text))
    assert r.emg.n == 4
    assert r.emg.table.blank_lines == [4]
    assert r.emg.raw_row(2)["sample_index"] == "2"
    assert any("blank line" in n for n in r.sources[r.emg.source.source_id].notes)


def test_duplicate_timestamps_gaps_and_segments_preserved(write, load):
    rows = []
    for i in range(10):                               # segment 1, idx 0..9 with a gap after 4
        idx = i if i < 5 else i + 3
        rows.append(emg_row(1, idx, 50 + idx / 1000, 1000.0 + (i // 4) * 0.1, 10.0 + (i // 4) * 0.1))
    rows.append(emg_row(1, 12, 50.012, 1000.3, 10.3))  # duplicate sample index 12
    for i in range(5):                                # segment 2 restarts at 0
        rows.append(emg_row(2, i, 90 + i / 1000, 1010.0, 20.0))
    r = load(write("d.csv", make_emg_text(rows)))
    emg = r.emg
    assert emg.n == 16
    assert [x.segment for x in emg.runs] == [1, 2]
    assert (emg.runs[0].start, emg.runs[0].stop) == (0, 11)
    # gap: no line across, duplicate index kept as its own row
    assert not emg.contiguous_after[4]
    assert list(emg.find_rows(12, segment=1)) == [9, 10]
    assert not emg.contiguous_after[10]               # end of segment run
    # identical receipt timestamps define a receipt group (rows 0..3)
    assert emg.receipt_group(2) == (0, 4)
    # duplicate sample index -> default axis falls back to original rows
    assert emg.default_axis() == AXIS_ROW


def test_non_contiguous_segment_values_are_separate_parts(write, load):
    rows = simple_emg_rows(4, seg=1) + simple_emg_rows(4, seg=2, start_idx=0) + \
        simple_emg_rows(4, seg=1, start_idx=100)
    r = load(write("p.csv", make_emg_text(rows)))
    runs = r.emg.runs
    assert [(x.segment, x.part) for x in runs] == [(1, 1), (2, 1), (1, 2)]
    assert any("separate row runs" in w for w in r.warnings)


def test_session_and_markers_recognized(write, workarea):
    e = write("s/emg.csv", make_emg_text(simple_emg_rows(20)))
    s = write("s/session.json", '{"session_id": "abc", "emg_rate": 1000.0}')
    m = write("s/markers.csv", 'host_time,host_mono,segment,emg_sample_index,label\n'
                               '1700000000.1,100.1,1,5,"a, quoted label"\n')
    write("s/notes.txt", "free text")
    out = import_paths([e.parent], workarea)          # folder import, only top level
    kinds = sorted(x.kind for x in out.sources)
    assert kinds == sorted([KIND_EMG, KIND_SESSION, KIND_MARKERS])
    mk = out.first(KIND_MARKERS)
    assert mk.labels == ["a, quoted label"]
    assert out.first(KIND_SESSION)["session_id"] == "abc"


def test_folder_import_does_not_recurse(write, workarea):
    write("f/emg.csv", make_emg_text(simple_emg_rows(5)))
    write("f/sub/contact.csv", CONTACT_HEADER + "\n1.0,1,1,1,1,1,1,1,1,1\n")
    out = import_paths([write("f/emg.csv", make_emg_text(simple_emg_rows(5))).parent], workarea)
    assert [s.kind for s in out.sources] == [KIND_EMG]
    assert any("not scanned" in n for n in out.notes)


def test_multiple_emg_recordings_are_ambiguous(write, workarea):
    a = write("a/emg.csv", make_emg_text(simple_emg_rows(5)))
    b = write("b/emg.csv", make_emg_text(simple_emg_rows(6)))
    out = import_paths([a, b], workarea)
    assert out.errors and "2 EMG recordings" in out.errors[0]


def test_contradictory_session_ids_rejected(write, workarea):
    e = write("a/emg.csv", make_emg_text(simple_emg_rows(5)))
    s1 = write("a/session.json", '{"session_id": "one"}')
    s2 = write("b/session.json", '{"session_id": "two"}')
    out = import_paths([e, s1, s2], workarea)
    assert any("contradictory session IDs" in x for x in out.errors)


def test_parse_csv_quoted_fields_keep_row_offsets(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text('a,label\n1,"multi\nline"\n2,plain\n', newline="")
    t = parse_csv(p, {"a": "int", "label": "text"})
    assert t.n_rows == 2
    assert t.texts["label"] == ["multi\nline", "plain"]
    assert t.raw_cells(1) == ["2", "plain"]
    assert t.raw_cells(0) == ["1", "multi\nline"]


def test_parse_csv_non_ascii_offsets(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("a,label\n1,größe\n2,ok\n", encoding="utf-8")
    t = parse_csv(p, {"a": "int"})
    assert t.raw_cells(1) == ["2", "ok"]
    assert t.raw_cells(0) == ["1", "größe"]


def test_duplicate_header_names_rejected(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("a,a\n1,2\n")
    with pytest.raises(CsvFormatError, match="duplicate"):
        parse_csv(p, {})
