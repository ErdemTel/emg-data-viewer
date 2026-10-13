"""Byte-for-byte source preservation through import, annotation edits, save and reopen."""

import json
import os
import shutil
import stat

import pytest

import emg_review.snapshot as snapshot_mod
from conftest import CONTACT_HEADER, fingerprint, make_emg_text, sha, simple_emg_rows
from emg_review.annotations import make_annotation
from emg_review.schema import KIND_CONTACT, KIND_EMG
from emg_review.snapshot import SnapshotError, snapshot_file
from emg_review.workspace import (Review, WorkspaceError, import_paths, open_review,
                                  save_review)


def _recording(write, folder="rec", n=64):
    e = write(f"{folder}/emg.csv", make_emg_text(simple_emg_rows(n)))
    c = write(f"{folder}/contact.csv", CONTACT_HEADER + "\n"
              "1700000000.000000,1,100000,200000,300000,400000,500000,600000,700000,800000\n")
    s = write(f"{folder}/session.json", json.dumps({"session_id": "s1", "emg_rate": 1000.0}))
    return e, c, s


def test_full_workflow_never_touches_originals_and_keeps_bytes(write, workarea, tmp_path):
    originals = _recording(write)
    before = {p: fingerprint(p) for p in originals}
    out = import_paths([originals[0].parent], workarea)
    review = Review.from_import(out)
    # snapshots are exact copies, read-only, inside the work area
    for sf in review.sources.values():
        assert sha(sf.snapshot_path) == sf.sha256 == sha(sf.original_path)
        assert workarea.owns(sf.snapshot_path)
        assert not os.stat(sf.snapshot_path).st_mode & stat.S_IWUSR
    # annotation edits
    a = make_annotation(review.emg, 3, 20, label="pre-grasp candidate")
    review.annotations.add(a)
    review.annotations.update(a.annotation_id, notes="edited", review_status="candidate")
    folder = save_review(review, tmp_path / "my_review", {"axis_mode": "row"})
    for sf in review.sources.values():
        copy = folder / "sources" / sf.source_id / sf.original_name
        assert copy.read_bytes() == open(sf.original_path, "rb").read()
    assert {p: fingerprint(p) for p in originals} == before
    # reopen after the original files are gone
    shutil.rmtree(originals[0].parent)
    workarea.cleanup()
    r2 = open_review(folder)
    assert r2.emg.n == 64 and r2.contact is not None and r2.session_meta["session_id"] == "s1"
    assert r2.annotations.items[0].notes == "edited"
    assert r2.view_state == {"axis_mode": "row"}
    assert not r2.dirty


def test_basename_collisions_get_separate_source_folders(write, workarea, tmp_path):
    e = write("a/data.csv", make_emg_text(simple_emg_rows(10)))
    c = write("b/data.csv", CONTACT_HEADER + "\n1700000000.0,1,1,2,3,4,5,6,7,8\n")
    out = import_paths([e, c], workarea)
    review = Review.from_import(out)
    paths = [sf.snapshot_path for sf in review.sources.values()]
    assert len({p.name for p in paths}) == 1 and len(set(paths)) == 2
    folder = save_review(review, tmp_path / "rv")
    stored = sorted((folder / "sources").glob("*/data.csv"))
    assert len(stored) == 2
    assert {sha(p) for p in stored} == {sha(e), sha(c)}
    r2 = open_review(folder)
    assert r2.emg is not None and r2.contact is not None


def test_source_changing_during_import_is_retried_then_rejected(write, tmp_path, monkeypatch):
    src = write("grow.csv", make_emg_text(simple_emg_rows(10)))
    real_copy = snapshot_mod.copy_verified

    def copy_then_append(s, d, expected_sha256=None):
        res = real_copy(s, d, expected_sha256)
        with open(src, "a") as f:                 # the recorder is still writing
            f.write("1,999,1,1,1,0,255,1,1,1,1,1,1,1,1\n")
        return res

    monkeypatch.setattr(snapshot_mod, "copy_verified", copy_then_append)
    with pytest.raises(SnapshotError, match="changed while it was being imported"):
        snapshot_file(src, tmp_path / "snap", retries=2, retry_delay_s=0)
    assert not list((tmp_path / "snap").iterdir())    # no inconsistent snapshot left behind


def test_source_changing_once_gives_consistent_snapshot_on_retry(write, tmp_path, monkeypatch):
    src = write("once.csv", make_emg_text(simple_emg_rows(10)))
    real_copy = snapshot_mod.copy_verified
    calls = []

    def copy_then_append_once(s, d, expected_sha256=None):
        res = real_copy(s, d, expected_sha256)
        if not calls:
            with open(src, "a") as f:
                f.write("1,999,1,1,1,0,255,1,1,1,1,1,1,1,1\n")
        calls.append(1)
        return res

    monkeypatch.setattr(snapshot_mod, "copy_verified", copy_then_append_once)
    info = snapshot_file(src, tmp_path / "snap", retries=2, retry_delay_s=0)
    assert len(calls) == 2
    assert info.path.read_bytes() == src.read_bytes()


def test_save_refuses_unrelated_or_foreign_folders(write, workarea, tmp_path):
    review = Review.from_import(import_paths([_recording(write)[0]], workarea))
    unrelated = tmp_path / "thesis"
    unrelated.mkdir()
    (unrelated / "chapter1.txt").write_text("keep me")
    with pytest.raises(WorkspaceError, match="not empty"):
        save_review(review, unrelated)
    assert (unrelated / "chapter1.txt").read_text() == "keep me"
    assert sorted(p.name for p in unrelated.iterdir()) == ["chapter1.txt"]
    other = Review.from_import(import_paths([_recording(write, "rec2")[0]], workarea))
    folder = save_review(other, tmp_path / "other_review")
    with pytest.raises(WorkspaceError, match="different review"):
        save_review(review, folder)
    # saving the same review again into its own folder is fine
    save_review(other, folder)


def test_reopen_detects_modified_sources(write, workarea, tmp_path):
    review = Review.from_import(import_paths([_recording(write)[0].parent], workarea))
    folder = save_review(review, tmp_path / "rv")
    contact = next(sf for sf in review.sources.values() if sf.kind == KIND_CONTACT)
    cpath = folder / "sources" / contact.source_id / contact.original_name
    os.chmod(cpath, 0o644)
    cpath.write_text(cpath.read_text().replace("100000", "100001"))
    r2 = open_review(folder)
    assert r2.contact is None
    assert any("does not match the recorded SHA-256" in w for w in r2.warnings)
    emg = next(sf for sf in review.sources.values() if sf.kind == KIND_EMG)
    epath = folder / "sources" / emg.source_id / emg.original_name
    os.chmod(epath, 0o644)
    with open(epath, "a") as f:
        f.write("1,9999,1,1,1,0,255,1,1,1,1,1,1,1,1\n")
    with pytest.raises(WorkspaceError, match="cannot be opened"):
        open_review(folder)


def test_resave_does_not_rewrite_sources(write, workarea, tmp_path):
    review = Review.from_import(import_paths([_recording(write)[0]], workarea))
    folder = save_review(review, tmp_path / "rv")
    sf = review.emg.source
    p = folder / "sources" / sf.source_id / sf.original_name
    st = os.stat(p)
    review.annotations.add(make_annotation(review.emg, 0, 1, annotation_type="bookmark"))
    save_review(review, folder)
    st2 = os.stat(p)
    assert (st.st_mtime_ns, st.st_ino) == (st2.st_mtime_ns, st2.st_ino)
    assert len(open_review(folder).annotations.items) == 1


def test_manifest_records_fingerprints(write, workarea, tmp_path):
    e, c, s = _recording(write)
    review = Review.from_import(import_paths([e, c, s], workarea))
    folder = save_review(review, tmp_path / "rv")
    m = json.loads((folder / "manifest.json").read_text())
    assert m["format"] == "emg-review-workspace"
    by_name = {x["original_filename"]: x for x in m["sources"]}
    assert by_name["emg.csv"]["sha256"] == sha(e)
    assert by_name["emg.csv"]["size_bytes"] == e.stat().st_size
    assert by_name["emg.csv"]["data_rows"] == 64
    assert len({x["source_id"] for x in m["sources"]}) == 3
    assert m["session_id"] == "s1"


def test_manifest_path_traversal_is_rejected(write, workarea, tmp_path):
    review = Review.from_import(import_paths([_recording(write)[0]], workarea))
    folder = save_review(review, tmp_path / "rv")
    m = json.loads((folder / "manifest.json").read_text())
    m["sources"][0]["path"] = "../../etc/passwd"
    (folder / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        open_review(folder)


def test_export_refuses_to_overwrite_a_source(write, workarea):
    from emg_review.annotations import export_annotations
    e = _recording(write)[0]
    review = Review.from_import(import_paths([e], workarea))
    before = sha(e)
    with pytest.raises(PermissionError):
        export_annotations([], e, review.protected_paths())
    assert sha(e) == before
