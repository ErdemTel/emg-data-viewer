"""Shared helpers. All test data is synthetic and generated at test time."""

import hashlib
import os
from pathlib import Path

import pytest

from emg_review.snapshot import WorkArea
from emg_review.workspace import import_paths, Review

EMG_HEADER = ("segment,sample_index,device_time,host_time,host_mono,lost,valid_mask,"
              "ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8")
CONTACT_HEADER = "host_time,segment,imp1,imp2,imp3,imp4,imp5,imp6,imp7,imp8"


def emg_row(seg, idx, dev, wall, mono, lost=0, mask=255, vals=None):
    vals = vals if vals is not None else [float(idx % 7 - 3)] * 8
    cells = ["" if v is None else repr(float(v)) for v in vals]
    return f"{seg},{idx},{dev:.6f},{wall:.6f},{mono:.6f},{lost},{mask}," + ",".join(cells)


def make_emg_text(rows):
    return EMG_HEADER + "\n" + "\n".join(rows) + "\n"


def simple_emg_rows(n=64, seg=1, start_idx=0, mono0=100.0, batch=16, wall0=1_700_000_000.0):
    """n rows, one segment; receipt groups of `batch` rows 30 ms apart; nominal 1000 Hz."""
    out = []
    for i in range(n):
        idx = start_idx + i
        mono = mono0 + (i // batch) * 0.030
        out.append(emg_row(seg, idx, 1_699_999_000.0 + idx / 1000.0, wall0 + mono - mono0, mono))
    return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(path):
    st = os.stat(path)
    return sha(path), st.st_size, st.st_mtime_ns


@pytest.fixture
def workarea(tmp_path):
    wa = WorkArea(tmp_path / "work")
    yield wa
    wa.cleanup()


@pytest.fixture
def write(tmp_path):
    def _write(rel, text, mode="w"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            p.write_bytes(text)
        else:
            p.write_text(text, newline="")
        return p
    return _write


@pytest.fixture
def load(workarea):
    """Import paths and build a Review (asserting the import had no fatal errors)."""
    def _load(*paths):
        out = import_paths(paths, workarea)
        assert not out.errors, out.errors
        return Review.from_import(out)
    return _load
