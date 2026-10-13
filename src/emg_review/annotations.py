"""User annotations (intervals and bookmarks), kept separate from the source data.

Row ranges are half-open over ORIGINAL data rows: [start_row, end_row_exclusive).
The inclusive last selected row is end_row_exclusive − 1. An annotation never crosses a
segment run. Endpoint sample indices and timestamps are copied as the exact original
text from the source file, to support later matching with other recordings — no
cross-modal alignment is implied. Labels are user assessments, not verified events.
"""

import csv
import io
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .recording import EmgRecording, now_iso
from .schema import N_CHANNELS, TIME_COLUMNS
from .snapshot import atomic_write_text

TYPE_INTERVAL = "interval"
TYPE_BOOKMARK = "bookmark"
REVIEW_STATUSES = ("unreviewed", "candidate", "reviewed", "rejected")
SOURCE_USER = "user"
SOURCE_DEMO = "synthetic-demo"


class SelectionError(ValueError):
    """An invalid row selection (empty, out of range, or crossing a segment run)."""


@dataclass
class Annotation:
    annotation_id: str
    annotation_type: str
    source_id: str
    source_sha256: str
    segment: str
    start_row: int
    end_row_exclusive: int
    start_sample_index: str
    end_sample_index_inclusive: str
    start_device_time: str = ""
    end_device_time_inclusive: str = ""
    start_host_time: str = ""
    end_host_time_inclusive: str = ""
    start_host_mono: str = ""
    end_host_mono_inclusive: str = ""
    selected_channels: str = "all"
    label: str = ""
    notes: str = ""
    annotation_source: str = SOURCE_USER
    review_status: str = "unreviewed"
    created: str = ""
    modified: str = ""

    @property
    def last_row(self) -> int:
        return self.end_row_exclusive - 1

    @property
    def n_rows(self) -> int:
        return self.end_row_exclusive - self.start_row

    def channel_list(self) -> list:
        """0-based channel indices."""
        if self.selected_channels.strip().lower() in ("", "all"):
            return list(range(N_CHANNELS))
        return [int(c) - 1 for c in self.selected_channels.split(";") if c.strip()]


ANNOTATION_FIELDS = [f.name for f in fields(Annotation)]
_INT_FIELDS = {"start_row", "end_row_exclusive"}


def channels_text(channels) -> str:
    """0-based channel indices -> 'all' or '1;3;8' (1-based, as in ch1..ch8)."""
    ch = sorted(set(int(c) for c in channels))
    if ch == list(range(N_CHANNELS)):
        return "all"
    return ";".join(str(c + 1) for c in ch)


def validate_range(emg: EmgRecording, start_row: int, end_row_exclusive: int):
    if isinstance(start_row, bool) or isinstance(end_row_exclusive, bool):
        raise SelectionError("row numbers must be integers")
    start_row, end_row_exclusive = int(start_row), int(end_row_exclusive)
    if end_row_exclusive <= start_row:
        raise SelectionError(f"empty selection [{start_row}, {end_row_exclusive})")
    if start_row < 0 or end_row_exclusive > emg.n:
        raise SelectionError(f"rows [{start_row}, {end_row_exclusive}) are outside 0..{emg.n - 1}")
    run = emg.run_of_row(start_row)
    if not run.contains(end_row_exclusive - 1):
        other = emg.run_of_row(end_row_exclusive - 1)
        raise SelectionError(f"selection would cross from {run.name} into {other.name}; "
                             "a selection must stay within one segment")
    return run


def make_annotation(emg: EmgRecording, start_row: int, end_row_exclusive: int, *,
                    annotation_type: str = TYPE_INTERVAL, label: str = "", notes: str = "",
                    channels=None, annotation_source: str = SOURCE_USER,
                    review_status: str = "unreviewed") -> Annotation:
    validate_range(emg, start_row, end_row_exclusive)
    if annotation_type == TYPE_BOOKMARK and end_row_exclusive != start_row + 1:
        raise SelectionError("a bookmark refers to exactly one row")
    first = emg.raw_row(start_row)
    last = emg.raw_row(end_row_exclusive - 1)
    now = now_iso()
    a = Annotation(
        annotation_id="ann-" + uuid.uuid4().hex[:12], annotation_type=annotation_type,
        source_id=emg.source.source_id, source_sha256=emg.source.sha256,
        segment=first.get("segment", ""), start_row=int(start_row),
        end_row_exclusive=int(end_row_exclusive),
        start_sample_index=first["sample_index"], end_sample_index_inclusive=last["sample_index"],
        selected_channels=channels_text(range(N_CHANNELS) if channels is None else channels),
        label=label, notes=notes, annotation_source=annotation_source,
        review_status=review_status, created=now, modified=now)
    for col in TIME_COLUMNS:
        if col in first:
            setattr(a, f"start_{col}", first[col])
            setattr(a, f"end_{col}_inclusive", last[col])
    return a


def check_annotation(a: Annotation, emg: EmgRecording) -> list:
    """Problems that make an annotation inconsistent with the loaded recording."""
    problems = []
    if a.source_sha256 != emg.source.sha256:
        problems.append(f"{a.annotation_id}: refers to source SHA-256 {a.source_sha256[:12]}…, "
                        f"loaded EMG is {emg.source.sha256[:12]}…")
        return problems
    try:
        validate_range(emg, a.start_row, a.end_row_exclusive)
    except SelectionError as e:
        problems.append(f"{a.annotation_id}: {e}")
        return problems
    if emg.raw_row(a.start_row)["sample_index"] != a.start_sample_index:
        problems.append(f"{a.annotation_id}: start sample index does not match row {a.start_row}")
    return problems


class AnnotationStore:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.dirty = False

    def add(self, a: Annotation):
        if any(x.annotation_id == a.annotation_id for x in self.items):
            raise ValueError(f"duplicate annotation id {a.annotation_id}")
        self.items.append(a)
        self.dirty = True

    def get(self, annotation_id: str) -> Annotation:
        for a in self.items:
            if a.annotation_id == annotation_id:
                return a
        raise KeyError(annotation_id)

    def update(self, annotation_id: str, **changes):
        a = self.get(annotation_id)
        editable = {"label", "notes", "review_status", "selected_channels"}
        bad = set(changes) - editable
        if bad:
            raise ValueError(f"not editable: {', '.join(sorted(bad))} (create a new annotation "
                             "to change the row range)")
        for k, v in changes.items():
            setattr(a, k, v)
        a.modified = now_iso()
        self.dirty = True

    def delete(self, annotation_id: str):
        a = self.get(annotation_id)
        self.items.remove(a)
        self.dirty = True

    def sorted(self) -> list:
        return sorted(self.items, key=lambda a: (a.start_row, a.end_row_exclusive, a.created))


def annotations_to_csv(items) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=ANNOTATION_FIELDS, lineterminator="\n")
    w.writeheader()
    for a in sorted(items, key=lambda a: (a.start_row, a.end_row_exclusive, a.annotation_id)):
        w.writerow(asdict(a))
    return buf.getvalue()


def annotations_from_csv(text: str) -> list:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    missing = [f for f in ("annotation_id", "source_sha256", "start_row", "end_row_exclusive")
               if f not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"annotations.csv is missing column(s): {', '.join(missing)}")
    out = []
    for i, row in enumerate(reader):
        kw = {}
        for f in ANNOTATION_FIELDS:
            v = row.get(f) or ""
            if f in _INT_FIELDS:
                try:
                    v = int(v)
                except ValueError:
                    raise ValueError(f"annotations.csv row {i}: {f} is not an integer: {v!r}")
            kw[f] = v
        out.append(Annotation(**kw))
    return out


def export_annotations(items, path: Path, protected_paths=()):
    """Write annotations to a CSV file. Refuses to write onto any imported source file."""
    path = Path(path)
    target = path.resolve()
    for p in protected_paths:
        if p and Path(p).resolve() == target:
            raise PermissionError(f"refusing to overwrite an imported source file: {path}")
    atomic_write_text(path, annotations_to_csv(items))
