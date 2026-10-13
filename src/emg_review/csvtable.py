"""Strict CSV reading that keeps the recorded sequence and stable row references.

- Data rows are numbered from 0 in file order (header excluded); rows are never
  reordered, deduplicated or dropped. A row with the wrong number of fields is an error.
- Blank lines are not data rows; they are reported (never silently skipped).
- Requested columns are converted to float64 (empty cell -> NaN, i.e. "not recorded") or
  int64 with a separate "known" mask. Unparseable cells are counted and kept visible via
  the exact original text, which can always be re-read from the (immutable) snapshot
  through the per-row byte offsets.
"""

import codecs
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHUNK_ROWS = 65536
MAX_EXAMPLES = 5
HEADER_PEEK_BYTES = 1 << 16


class CsvFormatError(ValueError):
    """Malformed, unreadable or unsupported CSV; the message is meant for the user."""


@dataclass
class BadCells:
    count: int = 0
    examples: list = field(default_factory=list)   # (data_row, original text)

    def record(self, row: int, text: str):
        self.count += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append((row, text))


@dataclass
class CsvTable:
    path: Path
    columns: list
    n_rows: int
    row_start: np.ndarray                 # byte offset of each data row in the file
    row_end: np.ndarray                   # byte offset just past the row
    floats: dict = field(default_factory=dict)
    ints: dict = field(default_factory=dict)
    int_known: dict = field(default_factory=dict)
    texts: dict = field(default_factory=dict)
    empty_cells: dict = field(default_factory=dict)
    bad_cells: dict = field(default_factory=dict)
    blank_lines: list = field(default_factory=list)   # 1-based physical line numbers

    def raw_cells(self, row: int) -> list:
        """Exact original cell texts of a data row, re-read from the file."""
        if not 0 <= row < self.n_rows:
            raise IndexError(f"data row {row} out of range 0..{self.n_rows - 1}")
        with open(self.path, "rb") as f:
            f.seek(int(self.row_start[row]))
            raw = f.read(int(self.row_end[row] - self.row_start[row]))
        return next(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))

    def raw_dict(self, row: int) -> dict:
        return dict(zip(self.columns, self.raw_cells(row)))

    def has(self, *names) -> bool:
        return all(n in self.columns for n in names)


def _decode(data: bytes, name: str) -> str:
    if b"\x00" in data:
        raise CsvFormatError(f"{name}: contains NUL bytes — not a text CSV file")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise CsvFormatError(f"{name}: not UTF-8 text (invalid byte at offset {e.start})") from e


def _header_columns(header: list, name: str) -> list:
    if not header:
        raise CsvFormatError(f"{name}: the first line is blank; expected a header row")
    columns = [h.strip() for h in header]
    for i, c in enumerate(columns):
        if not c:
            raise CsvFormatError(f"{name}: header column {i + 1} has no name")
    seen = set()
    for c in columns:
        if c in seen:
            raise CsvFormatError(f"{name}: duplicate column name '{c}' in header")
        seen.add(c)
    return columns


def read_header(path) -> list:
    """Column names of a CSV file (reads only the beginning of the file)."""
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read(HEADER_PEEK_BYTES)
    if not data.strip():
        raise CsvFormatError(f"{path.name}: file is empty")
    text = _decode(data, path.name)
    try:
        header = next(csv.reader(io.StringIO(text, newline="")))
    except StopIteration:
        raise CsvFormatError(f"{path.name}: file is empty")
    except csv.Error as e:
        raise CsvFormatError(f"{path.name}: malformed header: {e}") from e
    return _header_columns(header, path.name)


def _to_float(cells, name, row0, bad: BadCells):
    try:
        return np.array(cells, dtype=np.float64), 0
    except ValueError:
        pass
    out = np.empty(len(cells), dtype=np.float64)
    empty = 0
    for j, c in enumerate(cells):
        s = c.strip()
        if not s:
            out[j] = np.nan
            empty += 1
            continue
        try:
            out[j] = float(s)
        except ValueError:
            out[j] = np.nan
            bad.record(row0 + j, c)
    return out, empty


def _to_int(cells, name, row0, bad: BadCells):
    try:
        arr = np.array(cells, dtype=np.int64)
        return arr, np.ones(len(cells), dtype=bool), 0
    except (ValueError, OverflowError):
        pass
    out = np.zeros(len(cells), dtype=np.int64)
    known = np.zeros(len(cells), dtype=bool)
    empty = 0
    for j, c in enumerate(cells):
        s = c.strip()
        if not s:
            empty += 1
            continue
        try:
            out[j] = int(s)
            known[j] = True
        except (ValueError, OverflowError):
            bad.record(row0 + j, c)
    return out, known, empty


def parse_csv(path, types: dict, required: tuple = ()) -> CsvTable:
    """Parse a CSV file. types maps column name -> 'float' | 'int' | 'text' for the
    columns to convert (others stay available as raw text through raw_cells()).
    Columns in `required` must be present; required 'int' columns must also hold an
    integer in every data row. Otherwise CsvFormatError."""
    path = Path(path)
    name = path.name
    try:
        data = path.read_bytes()
    except OSError as e:
        raise CsvFormatError(f"{name}: cannot read file ({e.strerror or e})") from e
    if not data.strip():
        raise CsvFormatError(f"{name}: file is empty")
    base = len(codecs.BOM_UTF8) if data.startswith(codecs.BOM_UTF8) else 0
    text = _decode(data, name)
    del data
    is_ascii = text.isascii()

    starts, ends = [], []

    def lines():
        pos = base
        for line in io.StringIO(text, newline=""):
            starts.append(pos)
            pos += len(line) if is_ascii else len(line.encode("utf-8"))
            ends.append(pos)
            yield line

    reader = csv.reader(lines(), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise CsvFormatError(f"{name}: file is empty")
    except csv.Error as e:
        raise CsvFormatError(f"{name}: malformed header: {e}") from e
    columns = _header_columns(header, name)
    missing = [c for c in required if c not in columns]
    if missing:
        raise CsvFormatError(f"{name}: missing required column(s): {', '.join(missing)}")
    ncol = len(columns)
    conv = [(i, c, types[c]) for i, c in enumerate(columns) if c in types]
    parts = {c: [] for _, c, _ in conv}
    known_parts = {c: [] for _, c, k in conv if k == "int"}
    bad = {c: BadCells() for _, c, _ in conv}
    empties = {c: 0 for _, c, _ in conv}
    row_start, row_end, blank = [], [], []
    pending = []
    n_done = 0

    def flush():
        nonlocal n_done, pending
        if not pending:
            return
        for i, c, kind in conv:
            cells = [r[i] for r in pending]
            if kind == "float":
                arr, e = _to_float(cells, c, n_done, bad[c])
                parts[c].append(arr)
            elif kind == "int":
                arr, known, e = _to_int(cells, c, n_done, bad[c])
                parts[c].append(arr)
                known_parts[c].append(known)
                if c in required and not known.all():
                    j = int(np.flatnonzero(~known)[0])
                    raise CsvFormatError(
                        f"{name}: column '{c}' must hold an integer in every row; data row "
                        f"{n_done + j} has {cells[j]!r}")
            else:
                parts[c].append(cells)
                e = 0
            empties[c] += e
        n_done += len(pending)
        pending = []

    prev = reader.line_num
    try:
        for rec in reader:
            first, prev = prev, reader.line_num
            if not rec:
                blank.append(first + 1)
                continue
            if len(rec) != ncol:
                raise CsvFormatError(
                    f"{name}: data row {len(row_start)} (file line {first + 1}) has "
                    f"{len(rec)} fields but the header has {ncol}")
            row_start.append(starts[first])
            row_end.append(ends[prev - 1])
            pending.append(rec)
            if len(pending) >= CHUNK_ROWS:
                flush()
    except csv.Error as e:
        raise CsvFormatError(f"{name}: malformed CSV near file line {reader.line_num}: {e}") from e
    flush()

    table = CsvTable(path=path, columns=columns, n_rows=len(row_start),
                     row_start=np.asarray(row_start, dtype=np.int64),
                     row_end=np.asarray(row_end, dtype=np.int64),
                     empty_cells=empties, bad_cells={c: b for c, b in bad.items() if b.count},
                     blank_lines=blank)
    for _, c, kind in conv:
        if kind == "float":
            table.floats[c] = (np.concatenate(parts[c]) if parts[c]
                               else np.empty(0, dtype=np.float64))
        elif kind == "int":
            table.ints[c] = (np.concatenate(parts[c]) if parts[c]
                             else np.empty(0, dtype=np.int64))
            table.int_known[c] = (np.concatenate(known_parts[c]) if known_parts[c]
                                  else np.empty(0, dtype=bool))
        else:
            table.texts[c] = [s for chunk in parts[c] for s in chunk]
    return table
