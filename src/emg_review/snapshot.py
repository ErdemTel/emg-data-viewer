"""Byte-for-byte import snapshots, SHA-256 fingerprints and atomic metadata writes.

Importing a file copies its bytes into an app-managed work area; everything afterwards
reads the copy. The supplied file is only ever opened for reading. A source that
changes while it is being copied is retried and finally rejected, so a snapshot is
never a mix of two versions of a file.
"""

import hashlib
import os
import shutil
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20


class SnapshotError(Exception):
    """A source could not be snapshotted (unreadable, or changed during the copy)."""


def new_source_id() -> str:
    return "src-" + uuid.uuid4().hex[:12]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _stat_key(st: os.stat_result):
    return (st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev)


def make_read_only(path: Path):
    mode = os.stat(path).st_mode
    os.chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def copy_verified(src: Path, dst: Path, expected_sha256: str = None) -> tuple:
    """Copy src to a new file dst (never overwrites) and return (sha256, size) of the bytes
    written. The written file is re-read and must hash identically. If expected_sha256 is
    given the copy must match it. On any failure the partial copy is removed."""
    tmp = dst.with_name(f".{dst.name}.part-{uuid.uuid4().hex[:8]}")
    h = hashlib.sha256()
    size = 0
    try:
        with open(src, "rb") as fi, open(tmp, "xb") as fo:
            while chunk := fi.read(CHUNK):
                h.update(chunk)
                fo.write(chunk)
                size += len(chunk)
            fo.flush()
            os.fsync(fo.fileno())
        digest = h.hexdigest()
        if sha256_file(tmp) != digest:
            raise SnapshotError(f"copy of {src.name} does not read back identically")
        if expected_sha256 is not None and digest != expected_sha256:
            raise SnapshotError(f"{src.name}: SHA-256 {digest[:12]}… does not match the "
                                f"expected {expected_sha256[:12]}…")
        if dst.exists():
            raise SnapshotError(f"refusing to overwrite existing file {dst}")
        os.replace(tmp, dst)
        make_read_only(dst)
        return digest, size
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


@dataclass
class SnapshotInfo:
    path: Path          # the snapshot copy (read-only)
    sha256: str
    size: int
    source_mtime_ns: int


def snapshot_file(src: Path, dest_dir: Path, retries: int = 2, retry_delay_s: float = 0.25
                  ) -> SnapshotInfo:
    """Snapshot src into dest_dir/<src.name>. The source is stat'ed before and after the
    copy and re-hashed afterwards; any difference means it changed during import, and the
    copy is retried, then rejected with SnapshotError."""
    src = Path(src)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / src.name
    last_reason = ""
    for attempt in range(retries + 1):
        try:
            st0 = os.stat(src)
            if not stat.S_ISREG(st0.st_mode):
                raise SnapshotError(f"{src} is not a regular file")
            digest, size = copy_verified(src, dst)
            st1 = os.stat(src)
            source_digest = sha256_file(src)
        except PermissionError as e:
            raise SnapshotError(f"cannot read {src}: permission denied") from e
        except FileNotFoundError as e:
            raise SnapshotError(f"{src} no longer exists") from e
        except IsADirectoryError as e:
            raise SnapshotError(f"{src} is a directory") from e
        except OSError as e:
            raise SnapshotError(f"cannot read {src}: {e}") from e
        if _stat_key(st0) == _stat_key(st1) and source_digest == digest and size == st1.st_size:
            return SnapshotInfo(dst, digest, size, st1.st_mtime_ns)
        last_reason = ("size/modification time changed" if _stat_key(st0) != _stat_key(st1)
                       else "content changed")
        _force_remove(dst)
        if attempt < retries:
            time.sleep(retry_delay_s)
    raise SnapshotError(f"{src.name} changed while it was being imported ({last_reason}); "
                        f"tried {retries + 1} times. Wait until the file is complete "
                        "(e.g. the recording is stopped) and import again.")


def _force_remove(path: Path):
    try:
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
        path.unlink()
    except FileNotFoundError:
        pass


def _rmtree_onerror(func, path, exc_info):
    # snapshots are read-only; make them writable so our own work area can be removed
    try:
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        func(path)
    except OSError:
        pass


def remove_tree(path: Path):
    if path and Path(path).exists():
        shutil.rmtree(path, onerror=_rmtree_onerror)


class WorkArea:
    """App-managed temporary working area: <tmp>/emg-review-XXXX/sources/<source_id>/<name>.
    Removed when the app exits (cleanup())."""

    def __init__(self, root: Path = None):
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="emg-review-"))
        self.root.mkdir(parents=True, exist_ok=True)

    def new_source_dir(self, source_id: str) -> Path:
        d = self.root / "sources" / source_id
        d.mkdir(parents=True, exist_ok=False)
        return d

    def scratch_dir(self, prefix: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=prefix, dir=self.root))

    def owns(self, path: Path) -> bool:
        try:
            Path(path).resolve().relative_to(self.root.resolve())
            return True
        except ValueError:
            return False

    def discard(self, path: Path):
        """Remove a directory inside the work area (never anything outside it)."""
        if self.owns(path):
            remove_tree(path)

    def cleanup(self):
        remove_tree(self.root)


def atomic_write_text(path: Path, text: str):
    """Write text to path via a temporary file in the same directory + os.replace."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
