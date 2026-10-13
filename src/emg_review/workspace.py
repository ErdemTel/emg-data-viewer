"""Import pipeline, the in-memory Review, and saved review workspaces.

Saved workspace layout:

    review_folder/
      manifest.json                       review id, sources (sha256, size, names), notes
      sources/<source_id>/<original_name> unchanged byte copies of the imported files
      annotations.csv                     user annotations (separate from the data)
      view_state.json                     viewer settings

Workspace metadata is written atomically (temp file + rename); manifest.json last.
Source copies are verified against their SHA-256 when written and when reopened.
Reopening never uses the original import paths.
"""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .annotations import (AnnotationStore, annotations_from_csv, annotations_to_csv,
                          check_annotation)
from .csvtable import CsvFormatError, read_header
from .diagnostics import session_id
from .recording import (ContactTrack, EmgRecording, ImuTrack, MarkerTable, SourceFile, now_iso,
                        parse_contact, parse_emg, parse_imu, parse_markers, shared_receipt_clock)
from .schema import (KIND_CONTACT, KIND_EMG, KIND_IMU, KIND_LABELS, KIND_MARKERS, KIND_SESSION,
                     classify_csv_columns, classify_json, load_session_json)
from .snapshot import (SnapshotError, WorkArea, atomic_write_text, copy_verified, new_source_id,
                       sha256_file, snapshot_file)

REVIEW_FORMAT = "emg-review-workspace"
FORMAT_VERSION = 1
TIME_OVERLAP_TOLERANCE_S = 10.0
SNIFF_BYTES = 4096
FOLDER_SUFFIXES = {".csv", ".json"}

PARSERS = {KIND_EMG: parse_emg, KIND_CONTACT: parse_contact, KIND_IMU: parse_imu,
           KIND_MARKERS: parse_markers}


class WorkspaceError(Exception):
    pass


def is_review_folder(path: Path) -> bool:
    m = Path(path) / "manifest.json"
    if not m.is_file():
        return False
    try:
        return json.loads(m.read_text("utf-8")).get("format") == REVIEW_FORMAT
    except (OSError, ValueError, AttributeError):
        return False


# ------------------------------------------------------------------------------------------
# import
# ------------------------------------------------------------------------------------------

@dataclass
class ImportOutcome:
    sources: list = field(default_factory=list)        # SourceFile, recognized and parsed
    parsed: dict = field(default_factory=dict)         # source_id -> parsed object / dict
    rejected: list = field(default_factory=list)       # (name, reason)
    errors: list = field(default_factory=list)         # fatal: nothing should be imported
    notes: list = field(default_factory=list)
    review_folders: list = field(default_factory=list)  # dropped folders that are reviews
    snapshot_dirs: list = field(default_factory=list)

    def of_kind(self, kind) -> list:
        return [s for s in self.sources if s.kind == kind]

    def first(self, kind):
        s = self.of_kind(kind)
        return self.parsed[s[0].source_id] if s else None

    def discard(self, workarea: WorkArea):
        """Remove this outcome's snapshots from the work area (never anything else)."""
        for d in self.snapshot_dirs:
            workarea.discard(d)
        self.snapshot_dirs = []


def expand_paths(paths) -> tuple:
    """Files to import. A folder contributes only its own top-level .csv/.json files (no
    recursion, no other locations). Returns (files, notes, review_folders)."""
    files, notes, reviews = [], [], []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            if is_review_folder(p):
                reviews.append(p)
                continue
            found = []
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in FOLDER_SUFFIXES and not f.name.startswith("."):
                    found.append(f)
                elif f.is_dir():
                    notes.append(f"{p.name}/{f.name}/: subfolder not scanned")
            if not found:
                notes.append(f"{p}: no .csv/.json files in this folder")
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            notes.append(f"{p}: not found")
    # the same file dropped twice (e.g. file + its folder) is imported once
    unique, seen = [], set()
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique, notes, reviews


def _sniff_text(path: Path) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            head = f.read(SNIFF_BYTES)
    except OSError as e:
        return f"cannot read: {e.strerror or e}"
    if b"\x00" in head:
        return "binary file (not CSV/JSON text)"
    return None


def import_paths(paths, workarea: WorkArea) -> ImportOutcome:
    """Snapshot, recognize and parse the given files/folders. GUI-free; run it in a worker.
    The caller decides what to do with the outcome (new review / add companions) and must
    call outcome.discard(workarea) if it is not used."""
    out = ImportOutcome()
    files, notes, reviews = expand_paths(paths)
    out.notes.extend(notes)
    out.review_folders.extend(reviews)
    for f in files:
        why = _sniff_text(f)
        if why:
            out.rejected.append((f.name, why))
            continue
        sid = new_source_id()
        d = workarea.new_source_dir(sid)
        out.snapshot_dirs.append(d)
        try:
            snap = snapshot_file(f, d)
        except SnapshotError as e:
            out.rejected.append((f.name, str(e)))
            workarea.discard(d)
            out.snapshot_dirs.remove(d)
            continue
        sf = SourceFile(source_id=sid, kind=None, original_name=f.name,
                        original_path=str(f.resolve()), snapshot_path=snap.path,
                        sha256=snap.sha256, size=snap.size, imported_at=now_iso())
        kind, parsed, reason = _recognize_and_parse(sf)
        if kind is None:
            if reason.startswith("EMG:"):
                out.errors.append(f"{f.name}: {reason[4:].strip()}")
            out.rejected.append((f.name, reason))
            workarea.discard(d)
            out.snapshot_dirs.remove(d)
            continue
        sf.kind = kind
        if kind == KIND_SESSION:
            sf.columns = sorted(parsed.keys())
        else:
            sf.columns = list(parsed.table.columns)
            sf.n_rows = parsed.n
            if parsed.table.blank_lines:
                sf.notes.append(f"{len(parsed.table.blank_lines)} blank line(s) are not data rows "
                                f"(file lines {parsed.table.blank_lines[:5]})")
        out.sources.append(sf)
        out.parsed[sid] = parsed

    emgs = out.of_kind(KIND_EMG)
    if len(emgs) > 1:
        out.errors.append(
            f"{len(emgs)} EMG recordings were supplied together ("
            + ", ".join(_where(s) for s in emgs)
            + "). EMG Review handles one recording session at a time and never merges "
              "sessions — import one recording (file or folder) at a time.")
    for kind in (KIND_CONTACT, KIND_IMU, KIND_MARKERS, KIND_SESSION):
        same = out.of_kind(kind)
        if len(same) > 1:
            if kind == KIND_SESSION and len({session_id(out.parsed[s.source_id]) for s in same}) > 1:
                out.errors.append("contradictory session IDs among the supplied session files: "
                                  + ", ".join(f"{_where(s)}={session_id(out.parsed[s.source_id])}"
                                              for s in same))
                continue
            for s in same:
                out.rejected.append((s.original_name, f"ambiguous: {len(same)} "
                                     f"{KIND_LABELS[kind]} files supplied together "
                                     "(" + ", ".join(_where(x) for x in same) + ")"))
                out.sources.remove(s)
                out.parsed.pop(s.source_id)
                d = s.snapshot_path.parent
                workarea.discard(d)
                out.snapshot_dirs.remove(d)
    return out


def _where(sf: SourceFile) -> str:
    p = Path(sf.original_path)
    return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


def _recognize_and_parse(sf: SourceFile) -> tuple:
    """(kind, parsed, reason). Reasons starting with 'EMG:' are fatal EMG problems."""
    path = sf.snapshot_path
    try:
        with open(path, "rb") as f:
            head = f.read(SNIFF_BYTES)
    except OSError as e:
        return None, None, f"cannot read snapshot: {e}"
    looks_json = (path.suffix.lower() == ".json"
                  or head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"{"))
    if looks_json:
        try:
            doc = load_session_json(path)
        except ValueError as e:
            return None, None, str(e)
        kind, reason = classify_json(doc)
        return (kind, doc, reason) if kind else (None, None, reason)
    try:
        cols = read_header(path)
    except CsvFormatError as e:
        return None, None, str(e)
    kind, reason = classify_csv_columns(cols)
    if kind is None:
        if reason.startswith("looks like EMG"):
            return None, None, "EMG: insufficient required EMG columns — " + reason
        return None, None, reason
    try:
        parsed = PARSERS[kind](sf)
    except CsvFormatError as e:
        return None, None, ("EMG: " if kind == KIND_EMG else "") + str(e)
    return kind, parsed, reason


# ------------------------------------------------------------------------------------------
# consistency between an EMG recording and companions
# ------------------------------------------------------------------------------------------

def _span(track_times: dict, clock: str):
    t = track_times[clock]
    t = t[np.isfinite(t)]
    return (float(t.min()), float(t.max())) if len(t) else None


def consistency_issues(emg: EmgRecording, current_meta: Optional[dict], tracks: list,
                       new_meta: Optional[dict] = None) -> list:
    """Reasons to doubt that companions belong to this EMG recording. Never merges anything."""
    issues = []
    a, b = session_id(current_meta), session_id(new_meta)
    if a and b and a != b:
        issues.append(f"contradictory session IDs: loaded session {a}, new session file {b}")
    meta = new_meta or current_meta
    counts = (meta or {}).get("counts")
    if isinstance(counts, dict) and isinstance(counts.get("emg"), dict):
        written = counts["emg"].get("written")
        if isinstance(written, int) and written != emg.n:
            issues.append(f"session file says {written:,} EMG rows were written; the EMG file has "
                          f"{emg.n:,} data rows")
    emg_segs = set(np.unique(emg.segment).tolist()) if emg.has_segment else None
    for tr in tracks:
        name = tr.source.original_name
        if emg_segs is not None and tr.has_segment:
            segs = set(np.unique(tr.segment[tr.segment_known] if tr.segment_known is not None
                                 else tr.segment).tolist())
            extra = sorted(segs - emg_segs)
            if extra:
                issues.append(f"{name}: segment(s) {extra} do not exist in the EMG file")
        clock = shared_receipt_clock(emg, tr.times)
        if clock:
            s1, s2 = _span(emg.times, clock), _span(tr.times, clock)
            if s1 and s2 and (s2[1] < s1[0] - TIME_OVERLAP_TOLERANCE_S
                              or s2[0] > s1[1] + TIME_OVERLAP_TOLERANCE_S):
                issues.append(f"{name}: {clock} range does not overlap the EMG recording "
                              f"(likely a different recording)")
        if isinstance(tr, MarkerTable) and tr.emg_index is not None:
            missing = sum(1 for _, rows in tr.emg_rows(emg) if not rows)
            if missing:
                issues.append(f"{name}: {missing} marker(s) refer to EMG sample indices not "
                              "present in the EMG file")
    return issues


# ------------------------------------------------------------------------------------------
# the review
# ------------------------------------------------------------------------------------------

class Review:
    """One active recording session plus its companions, annotations and view state."""

    def __init__(self):
        self.review_id = "rev-" + uuid.uuid4().hex[:12]
        self.created = now_iso()
        self.sources = {}                 # source_id -> SourceFile
        self.emg: Optional[EmgRecording] = None
        self.contact: Optional[ContactTrack] = None
        self.imu: Optional[ImuTrack] = None
        self.markers: Optional[MarkerTable] = None
        self.session_meta: Optional[dict] = None
        self.annotations = AnnotationStore()
        self.view_state: dict = {}
        self.folder: Optional[Path] = None
        self.warnings: list = []
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty or self.annotations.dirty

    def mark_clean(self):
        self._dirty = False
        self.annotations.dirty = False

    def tracks(self) -> list:
        return [t for t in (self.contact, self.imu, self.markers) if t is not None]

    def protected_paths(self) -> list:
        """Every path holding imported data: snapshots, workspace copies, original files."""
        out = []
        for s in self.sources.values():
            out.append(s.snapshot_path)
            if s.original_path:
                out.append(Path(s.original_path))
            if self.folder:
                out.append(self.folder / "sources" / s.source_id / s.original_name)
        return out

    def _attach(self, sf: SourceFile, parsed):
        self.sources[sf.source_id] = sf
        if sf.kind == KIND_EMG:
            self.emg = parsed
        elif sf.kind == KIND_CONTACT:
            self.contact = parsed
        elif sf.kind == KIND_IMU:
            self.imu = parsed
        elif sf.kind == KIND_MARKERS:
            self.markers = parsed
        elif sf.kind == KIND_SESSION:
            self.session_meta = parsed

    @classmethod
    def from_import(cls, outcome: ImportOutcome) -> "Review":
        if outcome.errors:
            raise WorkspaceError("; ".join(outcome.errors))
        if not outcome.of_kind(KIND_EMG):
            raise WorkspaceError("no EMG recording among the supplied files")
        r = cls()
        for sf in outcome.sources:
            r._attach(sf, outcome.parsed[sf.source_id])
        r.warnings.extend(r.emg.warnings)
        r._dirty = True
        return r

    def companion_conflicts(self, outcome: ImportOutcome) -> list:
        """Kinds in the outcome that this review already has (v1 keeps one of each)."""
        have = {KIND_CONTACT: self.contact, KIND_IMU: self.imu, KIND_MARKERS: self.markers,
                KIND_SESSION: self.session_meta}
        out = []
        for sf in outcome.sources:
            if sf.kind == KIND_EMG:
                out.append(f"{sf.original_name}: an EMG recording is already loaded")
            elif have.get(sf.kind) is not None:
                same = [s for s in self.sources.values() if s.kind == sf.kind and s.sha256 == sf.sha256]
                out.append(f"{sf.original_name}: {'identical file already imported' if same else 'a ' + KIND_LABELS[sf.kind] + ' file is already loaded'}")
        return out

    def add_companions(self, outcome: ImportOutcome):
        conflicts = self.companion_conflicts(outcome)
        if conflicts:
            raise WorkspaceError("; ".join(conflicts))
        for sf in outcome.sources:
            self._attach(sf, outcome.parsed[sf.source_id])
        self._dirty = True


def import_issues(outcome: ImportOutcome, review: Optional[Review] = None) -> list:
    """Consistency issues for a new review (review=None) or for adding companions."""
    if review is None:
        emg = outcome.first(KIND_EMG)
        if emg is None:
            return []
        tracks = [outcome.first(k) for k in (KIND_CONTACT, KIND_IMU, KIND_MARKERS)]
        return consistency_issues(emg, None, [t for t in tracks if t is not None],
                                  outcome.first(KIND_SESSION))
    tracks = [outcome.first(k) for k in (KIND_CONTACT, KIND_IMU, KIND_MARKERS)]
    return consistency_issues(review.emg, review.session_meta, [t for t in tracks if t is not None],
                              outcome.first(KIND_SESSION))


# ------------------------------------------------------------------------------------------
# save / open
# ------------------------------------------------------------------------------------------

def _check_target(review: Review, folder: Path):
    if folder.exists():
        if not folder.is_dir():
            raise WorkspaceError(f"{folder} exists and is not a folder")
        manifest = folder / "manifest.json"
        if manifest.exists():
            try:
                m = json.loads(manifest.read_text("utf-8"))
            except (OSError, ValueError) as e:
                raise WorkspaceError(f"{folder} has an unreadable manifest.json ({e}); "
                                     "refusing to overwrite") from e
            if m.get("format") != REVIEW_FORMAT:
                raise WorkspaceError(f"{folder} is not an EMG Review workspace; refusing to overwrite")
            if m.get("review_id") != review.review_id:
                raise WorkspaceError(f"{folder} holds a different review ({m.get('review_id')}); "
                                     "choose a new or empty folder")
        elif any(folder.iterdir()):
            raise WorkspaceError(f"{folder} is not empty and is not this review's workspace; "
                                 "choose a new or empty folder")


def save_review(review: Review, folder, view_state: Optional[dict] = None) -> Path:
    folder = Path(folder).absolute()
    _check_target(review, folder)
    folder.mkdir(parents=True, exist_ok=True)
    for sf in review.sources.values():
        dest_dir = folder / "sources" / sf.source_id
        dest = dest_dir / sf.original_name
        if dest.exists():
            if dest.stat().st_size != sf.size or sha256_file(dest) != sf.sha256:
                raise WorkspaceError(f"{dest} exists but differs from the imported source "
                                     f"{sf.original_name}; refusing to overwrite it")
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        copy_verified(sf.snapshot_path, dest, expected_sha256=sf.sha256)
    if view_state is not None:
        review.view_state = dict(view_state)
    atomic_write_text(folder / "annotations.csv", annotations_to_csv(review.annotations.items))
    atomic_write_text(folder / "view_state.json", json.dumps(review.view_state, indent=2))
    manifest = {
        "format": REVIEW_FORMAT,
        "format_version": FORMAT_VERSION,
        "review_id": review.review_id,
        "app": "EMG Review",
        "app_version": __version__,
        "created": review.created,
        "saved": now_iso(),
        "session_id": session_id(review.session_meta),
        "emg_source_id": review.emg.source.source_id if review.emg else None,
        "sources": [sf.manifest_entry() for sf in review.sources.values()],
        "files": {
            "sources/": "unchanged byte copies of the imported files (verify with sha256)",
            "annotations.csv": "user annotations; half-open original row ranges "
                               "[start_row, end_row_exclusive)",
            "view_state.json": "viewer settings only",
        },
    }
    atomic_write_text(folder / "manifest.json", json.dumps(manifest, indent=2))
    review.folder = folder
    review.mark_clean()
    return folder


def open_review(folder) -> Review:
    """Open a saved workspace. Every source is checked against its recorded size and
    SHA-256; the EMG source must verify, a failing companion is skipped with a warning."""
    folder = Path(folder).absolute()
    mpath = folder / "manifest.json"
    try:
        m = json.loads(mpath.read_text("utf-8"))
    except FileNotFoundError:
        raise WorkspaceError(f"{folder} has no manifest.json — not an EMG Review workspace")
    except (OSError, ValueError) as e:
        raise WorkspaceError(f"cannot read {mpath}: {e}") from e
    if m.get("format") != REVIEW_FORMAT:
        raise WorkspaceError(f"{folder} is not an EMG Review workspace")
    if int(m.get("format_version", 0)) > FORMAT_VERSION:
        raise WorkspaceError(f"workspace format {m.get('format_version')} is newer than this app supports")
    review = Review()
    review.review_id = m.get("review_id") or review.review_id
    review.created = m.get("created", review.created)
    root = folder.resolve()
    for e in m.get("sources", []):
        kind = e.get("kind")
        rel = e.get("path", "")
        p = (folder / rel).resolve()
        label = f"{e.get('original_filename')} ({e.get('source_id')})"
        problem = None
        try:
            p.relative_to(root)
        except ValueError:
            problem = f"{label}: path {rel!r} points outside the workspace"
        if problem is None and not p.is_file():
            problem = f"{label}: missing from the workspace ({rel})"
        if problem is None and (p.stat().st_size != e.get("size_bytes")
                                or sha256_file(p) != e.get("sha256")):
            problem = f"{label}: content does not match the recorded SHA-256 — file was modified"
        if problem:
            if kind == KIND_EMG:
                raise WorkspaceError(problem + "; the review cannot be opened")
            review.warnings.append(problem + "; skipped")
            continue
        sf = SourceFile(source_id=e["source_id"], kind=kind, original_name=e["original_filename"],
                        original_path=e.get("original_path_at_import", ""), snapshot_path=p,
                        sha256=e["sha256"], size=e["size_bytes"], imported_at=e.get("imported_at", ""),
                        columns=e.get("columns", []), n_rows=e.get("data_rows"),
                        notes=e.get("notes", []))
        try:
            parsed = load_session_json(p) if kind == KIND_SESSION else PARSERS[kind](sf)
        except (CsvFormatError, ValueError, KeyError) as ex:
            if kind == KIND_EMG:
                raise WorkspaceError(f"{label}: {ex}") from ex
            review.warnings.append(f"{label}: {ex}; skipped")
            continue
        review._attach(sf, parsed)
    if review.emg is None:
        raise WorkspaceError("the workspace has no EMG source")
    review.warnings.extend(review.emg.warnings)
    ann = folder / "annotations.csv"
    if ann.exists():
        try:
            items = annotations_from_csv(ann.read_text("utf-8"))
        except (OSError, ValueError) as ex:
            raise WorkspaceError(f"cannot read annotations.csv: {ex}") from ex
        for a in items:
            review.warnings.extend(check_annotation(a, review.emg))
        review.annotations = AnnotationStore(items)
    vs = folder / "view_state.json"
    if vs.exists():
        try:
            review.view_state = json.loads(vs.read_text("utf-8"))
        except (OSError, ValueError) as ex:
            review.warnings.append(f"view_state.json unreadable ({ex}); default view used")
    review.folder = folder
    review.mark_clean()
    return review
