"""The render cache: the scene assets camera images are drawn from, on this
machine, encrypted, kept between sessions.

The hosting platform sends every scene asset encrypted (DAC1, see
:mod:`world_engine_client.dac1`) under its key for that scene: your account's
own, or one shared by all the accounts that may read the scene. This client
stores the bytes exactly as they arrive and asks the platform for the key at
every connect; the key lives in memory only. A scene made private, a revoked
key or a new key for the scene (a new key epoch, say) therefore makes what is
cached unreadable at once, and the platform's daily download limits count
only real downloads.

Where (``DIGIARENA_CACHE_DIR`` overrides; directories are 0700, files 0600)::

    <user cache dir>/digiarena/render/
        index.sqlite3        scenes, files, when each was last used, settings
        blobs/ab/<name>      one file per asset: DAC1 as received, named by an
                             HMAC of its digest under the scene key; or a plain
                             public file (robot meshes from MuJoCo Menagerie, or
                             a scene of your own local ``world-engine serve``)
        pages/ab/<name>      the capture page's scripts, by server and by their
                             content-hashed path
        tmp/<name>.<pid>.<random>.part   downloads in progress
        locks/<name>.lock    one lock per file; the operating system releases it
                             when the process holding it dies

The user cache dir is ``$XDG_CACHE_HOME`` or ``~/.cache`` on Linux,
``~/Library/Caches`` on macOS and ``%LOCALAPPDATA%`` on Windows.

Retention: a file not used for ``max_age_days`` (default 30; 0 keeps files
however old) is deleted, unless its scene is pinned. ``max_size_gb`` (default:
no limit) deletes the least recently used files of unpinned scenes once the
cache is larger. The check runs in the background when a renderer starts, at
most once a day, after that session's files are in place: the scene being
opened is not deleted. ``DIGIARENA_CACHE_MAX_AGE_DAYS`` and
``DIGIARENA_CACHE_MAX_SIZE_GB`` override the stored settings.

API: :func:`list_scenes` (also ``list``), :func:`pin`, :func:`unpin`,
:func:`clear`, :func:`configure`, :func:`settings`, :func:`cleanup`. The same
from a shell: ``python -m world_engine_client.cache ls|pin|unpin|clear|config``
(``digiarena cache ...`` with the DigiArena SDK).

What the encryption protects against: someone with access to this disk but not
to an account that may read the scene (for a public scene, any account), and
casual browsing of the cache directory. It does not protect against you: your
account can read the scene, so your key decrypts it, and the images and the
decrypted assets exist in the memory of this process, of headless Chrome and
of the GPU while a session runs.

:class:`SceneAssets` is the part a renderer uses: it fetches the key, makes
sure every asset of a session is in the cache and verified, and hands out the
plaintext in memory. Both the Chrome capture page path and a native renderer
read assets through it.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import errno
import hashlib
import itertools
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import dac1
from .errors import RendererError, RenderFetchError

log = logging.getLogger("world_engine_client.cache")

DEFAULT_MAX_AGE_DAYS = 30.0
_DAY = 86400.0
_HOUR = 3600.0
_CLEANUP_EVERY = _DAY
# A file in use stays at least this long, whatever the settings: a session
# touches its files when it starts and may read them again later.
_IN_USE = _HOUR
# Files under a key this scene no longer has, unused for this long, go when
# the scene's new key arrives (the old ones can never be decrypted again).
_OLD_KEY_GRACE = _DAY
_STALE_TEMP = _DAY  # leftover we_render_* directories of old clients, *.part files
_RETRIES = 3  # resumed downloads after a dropped connection
# A process waiting for another to finish a download stops waiting when the
# other's download has not grown for this long, and downloads it itself.
_STALL_SECONDS = 120.0
_KEY_REPLY_MAX = 64 * 1024
_SCHEMA = 1

# Network file systems: SQLite's WAL mode needs every process on one machine.
_NETWORK_FS = {
    "nfs",
    "nfs4",
    "lustre",
    "gpfs",
    "cifs",
    "smb3",
    "smbfs",
    "ceph",
    "glusterfs",
    "fuse.glusterfs",
    "fuse.sshfs",
    "fuse.s3fs",
    "fuse.gcsfuse",
    "fuse.ceph-fuse",
    "fuse.juicefs",
    "9p",
    "afs",
    "beegfs",
    "panfs",
}

# The capture page's scripts have Vite's content hash in their names: one name,
# one content, so they can be kept. index.html has no hash and is fetched fresh.
_HASHED_PAGE_FILE = re.compile(r"/assets/[A-Za-z0-9_.-]+-[A-Za-z0-9_-]{8}\.[A-Za-z0-9]+")
_ASSET_URL = re.compile(r"/[A-Za-z0-9_.~%/-]+")

_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS scene ("
    " scope TEXT PRIMARY KEY,"
    " display_name TEXT NOT NULL,"
    " named INTEGER NOT NULL DEFAULT 0,"  # 1: the caller gave the name
    " encrypted INTEGER NOT NULL DEFAULT 0,"
    " pinned INTEGER NOT NULL DEFAULT 0,"
    " key_id TEXT,"
    " created REAL NOT NULL,"
    " last_used REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS blob ("
    " name TEXT PRIMARY KEY,"
    " kind TEXT NOT NULL,"  # dac1 | plain | page
    " size INTEGER NOT NULL,"
    " key_id TEXT,"
    " sha256 TEXT,"  # page files only: their content hash
    " created REAL NOT NULL,"
    " last_used REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS scene_blob ("
    " scope TEXT NOT NULL, name TEXT NOT NULL, PRIMARY KEY (scope, name))",
    "CREATE INDEX IF NOT EXISTS scene_blob_name ON scene_blob (name)",
)

_UNSET = object()
_warned: set[str] = set()
_warn_lock = threading.Lock()


def _warn_once(key: str, message: str) -> None:
    with _warn_lock:
        if key in _warned:
            return
        _warned.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


# --------------------------------------------------------------------------- #
# Where
# --------------------------------------------------------------------------- #
def user_cache_dir() -> Path:
    """The per-user cache directory of this platform."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        return Path(local) if local else Path.home() / "AppData" / "Local"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".cache"


def directory(cache_dir: str | os.PathLike | None = None) -> Path:
    """The render cache directory: ``cache_dir``, else ``DIGIARENA_CACHE_DIR``,
    else ``<user cache dir>/digiarena/render``."""
    if cache_dir:
        return Path(cache_dir).expanduser()
    env = os.environ.get("DIGIARENA_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return user_cache_dir() / "digiarena" / "render"


def _private_dir(path: Path) -> None:
    """Create ``path`` (and missing parents) readable by this user only, and
    make an existing one private when this user owns it."""
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        if p.parent == p:
            break
        p = p.parent
    for p in reversed(missing):
        with contextlib.suppress(FileExistsError):
            os.mkdir(p, 0o700)
    if os.name != "nt":
        st = path.stat()
        if st.st_uid == os.getuid() and st.st_mode & 0o077:
            os.chmod(path, 0o700)


def _fs_type(path: Path) -> str | None:
    """The file system type of ``path`` on Linux (from /proc/self/mountinfo)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        real = os.path.realpath(path)
        best, kind = "", None
        with open("/proc/self/mountinfo") as f:
            for line in f:
                left, _, right = line.partition(" - ")
                fields = left.split()
                if len(fields) < 5 or not right:
                    continue
                mount = fields[4]
                for code, char in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n")):
                    mount = mount.replace(code, char)
                mount = mount.replace("\\134", "\\")
                inside = real == mount or real.startswith(mount.rstrip("/") + "/")
                if inside and len(mount) >= len(best):
                    best, kind = mount, right.split()[0]
        return kind
    except OSError:
        return None


def _utc(ts: float | None) -> datetime | None:
    return None if ts is None else datetime.fromtimestamp(ts, UTC)


def _pid_alive(pid: int) -> bool:
    """Whether a process of this machine runs (POSIX). On Windows os.kill(pid,
    0) would end the process, so there every process counts as running."""
    if os.name == "nt":
        return True
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


# --------------------------------------------------------------------------- #
# Locks: one per file name, across threads and processes
# --------------------------------------------------------------------------- #
_thread_locks: dict[tuple[str, str], threading.Lock] = {}
_thread_locks_guard = threading.Lock()


def _thread_lock(root: Path, name: str) -> threading.Lock:
    key = (str(root), name)
    with _thread_locks_guard:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = _thread_locks[key] = threading.Lock()
        return lock


class _NameLock:
    """Exclusive use of one cache file name. A thread lock covers this process
    (POSIX locks on some network file systems do not separate threads); an
    OS file lock covers other processes and is released by the kernel when
    the holder dies, so a download killed with SIGKILL never blocks anyone.
    Where the file system has no locks, only this process is covered: two
    processes may then both download a file, which is still safe (unique
    temporary names, verified content, atomic rename)."""

    def __init__(self, cache: RenderCache, name: str):
        self._cache, self._name = cache, name
        self._path = cache.locks / f"{name}.lock"
        self._thread = _thread_lock(cache.root, name)
        self._fd: int | None = None
        self._has_thread = False

    def acquire(self, give_up: Callable[[], bool] | None = None, *, wait: bool = True) -> bool:
        """Block until held. ``give_up()`` is asked while waiting: True stops
        the wait and returns False. ``wait=False`` tries once."""
        delay = 0.02
        if wait:
            while not self._thread.acquire(timeout=delay):
                if give_up is not None and give_up():
                    return False
                delay = min(delay * 1.5, 0.5)
        elif not self._thread.acquire(blocking=False):
            return False
        self._has_thread = True
        try:
            while not self._try_file_lock():
                if not wait or (give_up is not None and give_up()):
                    self._release_thread()
                    return False
                time.sleep(delay)
                delay = min(delay * 1.5, 0.5)
            return True
        except BaseException:
            self._release_thread()
            raise

    def _try_file_lock(self) -> bool:
        if not self._cache.file_locks:
            return True
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if not _lock_fd(fd):
                os.close(fd)
                return False
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL):
                self._cache.file_locks = False
                _warn_once(
                    f"locks:{self._cache.root}",
                    f"the render cache {self._cache.root} is on a file system without file "
                    "locks: processes sharing it may download a file twice. Set "
                    "DIGIARENA_CACHE_DIR to a directory on a local disk",
                )
                return True
            raise
        if os.name != "nt":
            # Cleanup may have removed this lock file after we opened it: a
            # lock on a file no longer at the path excludes nobody.
            try:
                same = os.fstat(fd).st_ino == os.stat(self._path).st_ino
            except FileNotFoundError:
                same = False
            if not same:
                _unlock_fd(fd)
                os.close(fd)
                return False
        self._fd = fd
        return True

    def _release_thread(self) -> None:
        if self._has_thread:
            self._has_thread = False
            self._thread.release()

    def release(self, *, remove: bool = False) -> None:
        """Release; ``remove`` also deletes the lock file (safe while held:
        a waiter that opened it notices and opens the path again)."""
        fd, self._fd = self._fd, None
        if fd is not None:
            if remove and os.name != "nt":
                with contextlib.suppress(OSError):
                    os.unlink(self._path)
            with contextlib.suppress(OSError):
                _unlock_fd(fd)
            os.close(fd)
        self._release_thread()

    def __enter__(self) -> _NameLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _lock_fd(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
        return True

    def _unlock_fd(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_fd(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #
class CacheUnavailable(OSError):
    """The cache directory cannot be used (not writable, not a directory, a
    broken index)."""


class RenderCache:
    """One cache directory: its files, index and settings. Safe to use from
    several threads and processes at once."""

    def __init__(self, root: str | os.PathLike | None = None):
        self.root = directory(root)
        self.blobs = self.root / "blobs"
        self.pages = self.root / "pages"
        self.tmp = self.root / "tmp"
        self.locks = self.root / "locks"
        self.index = self.root / "index.sqlite3"
        self.file_locks = True
        self.network_fs = False
        self._local = threading.local()  # .con: this thread's connection inside batch()
        try:
            for d in (self.root, self.blobs, self.pages, self.tmp, self.locks):
                _private_dir(d)
            if not self.index.exists():
                with contextlib.suppress(FileExistsError):
                    os.close(os.open(self.index, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
            fs = _fs_type(self.root)
            self.network_fs = fs in _NETWORK_FS
            if self.network_fs:
                _warn_once(
                    f"nfs:{self.root}",
                    f"the render cache {self.root} is on {fs}, a network file system. Set "
                    "DIGIARENA_CACHE_DIR to a directory on a local disk: machines sharing one "
                    "cache are slow and may download files twice",
                )
            # Processes that open a new cache at the same moment would both
            # switch its journal mode, and SQLite answers one of them "database
            # is locked" at once, without waiting: one at a time.
            lock = _NameLock(self, ".index")
            deadline = time.monotonic() + 30
            held = lock.acquire(give_up=lambda: time.monotonic() > deadline)
            try:
                self._init_index()
            finally:
                if held:
                    lock.release()
        except (OSError, sqlite3.Error) as exc:
            raise CacheUnavailable(f"the render cache {self.root} cannot be used: {exc}") from exc

    def _init_index(self) -> None:
        want = "delete" if self.network_fs else "wal"
        for attempt in range(6):
            try:
                with self._db() as con:
                    if con.execute("PRAGMA journal_mode").fetchone()[0].lower() != want:
                        con.execute(f"PRAGMA journal_mode={want}")
                with self._db(write=True) as con:
                    for statement in _SCHEMA_SQL:
                        con.execute(statement)
                    con.execute(
                        "INSERT OR IGNORE INTO meta (k, v) VALUES ('schema', ?)", (str(_SCHEMA),)
                    )
                return
            except sqlite3.OperationalError as exc:
                text = str(exc).lower()
                if attempt == 5 or not ("locked" in text or "busy" in text):
                    raise
                time.sleep(0.05 * 2**attempt)

    # ----- index ------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.index), timeout=30, isolation_level=None)
        try:
            con.execute("PRAGMA busy_timeout = 30000")
            if con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
                # A commit no longer waits for the disk. In WAL mode a crash
                # may lose the last commits but never breaks the index, and
                # the files are checked when used anyway.
                con.execute("PRAGMA synchronous = NORMAL")
        except BaseException:
            con.close()
            raise
        return con

    @contextlib.contextmanager
    def batch(self) -> Iterator[None]:
        """One index connection for this thread until the block ends. Each
        statement on a connection of its own cost about 12 ms, most of it in
        the checkpoint SQLite runs when the last connection closes: a session
        start grew with its number of files. Not kept beyond the block, since
        an open SQLite connection must not be carried across fork()."""
        if getattr(self._local, "con", None) is not None:
            yield
            return
        con = self._connect()
        self._local.con = con
        try:
            yield
        finally:
            self._local.con = None
            con.close()

    @contextlib.contextmanager
    def _db(self, write: bool = False):
        con = getattr(self._local, "con", None)
        own = con is None
        if own:
            con = self._connect()
        try:
            if write:
                con.execute("BEGIN IMMEDIATE")
            try:
                yield con
                if write:
                    con.execute("COMMIT")
            except BaseException:
                # A connection kept for a batch must not stay in a transaction.
                if write and con.in_transaction:
                    with contextlib.suppress(sqlite3.Error):
                        con.execute("ROLLBACK")
                raise
        finally:
            if own:
                con.close()

    @staticmethod
    def _meta(con, key: str) -> str | None:
        row = con.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    # ----- settings --------------------------------------------------------- #
    def settings(self) -> dict:
        """The settings in force: environment variables first, then what
        :func:`configure` stored, then the defaults."""
        with self._db() as con:
            stored_age = self._meta(con, "max_age_days")
            stored_size = self._meta(con, "max_size_gb")
        age = DEFAULT_MAX_AGE_DAYS if stored_age is None else float(stored_age)
        size = None if stored_size in (None, "") else float(stored_size)
        overridden = []
        env_age = os.environ.get("DIGIARENA_CACHE_MAX_AGE_DAYS")
        if env_age not in (None, ""):
            age = _check_age(env_age, "DIGIARENA_CACHE_MAX_AGE_DAYS")
            overridden.append("max_age_days")
        env_size = os.environ.get("DIGIARENA_CACHE_MAX_SIZE_GB")
        if env_size not in (None, ""):
            size = _check_size(env_size, "DIGIARENA_CACHE_MAX_SIZE_GB")
            overridden.append("max_size_gb")
        return {
            "dir": str(self.root),
            "max_age_days": age,
            "max_size_gb": size,
            "from_environment": overridden,
        }

    def configure(self, *, max_age_days=_UNSET, max_size_gb=_UNSET) -> dict:
        updates = {}
        if max_age_days is not _UNSET:
            updates["max_age_days"] = repr(_check_age(max_age_days, "max_age_days"))
        if max_size_gb is not _UNSET:
            size = _check_size(max_size_gb, "max_size_gb")
            updates["max_size_gb"] = "" if size is None else repr(size)
        if updates:
            with self._db(write=True) as con:
                for k, v in updates.items():
                    con.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (k, v))
        return self.settings()

    # ----- files ------------------------------------------------------------ #
    def blob_path(self, name: str) -> Path:
        return self.blobs / name[:2] / name

    def page_path(self, name: str) -> Path:
        return self.pages / name[:2] / name

    def path_of(self, name: str, kind: str) -> Path:
        return self.page_path(name) if kind == "page" else self.blob_path(name)

    def lock(self, name: str) -> _NameLock:
        return _NameLock(self, name)

    def new_part(self, name: str) -> tuple[Path, int]:
        """A new temporary file for a download of ``name``: 0600, a name no
        other download uses."""
        path = self.tmp / f"{name}.{os.getpid()}.{secrets.token_hex(4)}.part"
        return path, os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)

    def parts(self, name: str) -> list[Path]:
        return sorted(self.tmp.glob(f"{name}.*.part"))

    def remove_parts(self, name: str) -> None:
        """Delete the leftovers of earlier downloads of ``name``. Only called
        with the name's lock held, so no download that locked owns them."""
        for part in self.parts(name):
            if not self._part_is_live(part):
                with contextlib.suppress(OSError):
                    part.unlink()

    def install(self, part: Path, name: str, kind: str, check: Callable[[Path], None]) -> Path:
        """Move a finished download into place. Another process may have put
        the same file there meanwhile (it downloaded without the lock): any
        copy that verifies is a copy of the same content."""
        final = self.path_of(name, kind)
        _private_dir(final.parent)
        for attempt in range(5):
            try:
                os.replace(part, final)
                return final
            except PermissionError:  # Windows: the target is open elsewhere
                if final.exists():
                    try:
                        check(final)
                    except Exception:  # noqa: BLE001 - not a good copy: try again
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    with contextlib.suppress(OSError):
                        part.unlink()
                    return final
                time.sleep(0.2 * (attempt + 1))
        os.replace(part, final)
        return final

    # ----- rows ------------------------------------------------------------- #
    def touch_scene(
        self, scope: str, display_name: str, *, named: bool, encrypted: bool, key_id=None
    ) -> None:
        """Record a session of a scene. A name the caller gave replaces any
        earlier one; a fallback name (the engine's scene id) only fills in."""
        now = time.time()
        with self._db(write=True) as con:
            row = con.execute("SELECT named FROM scene WHERE scope = ?", (scope,)).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO scene (scope, display_name, named, encrypted, pinned, key_id,"
                    " created, last_used) VALUES (?, ?, ?, ?, 0, ?, ?, ?)",
                    (scope, display_name, int(named), int(encrypted), key_id, now, now),
                )
                return
            rename = named or not row[0]
            con.execute(
                "UPDATE scene SET last_used = ?, encrypted = ?, key_id = COALESCE(?, key_id)"
                " WHERE scope = ?",
                (now, int(encrypted), key_id, scope),
            )
            if rename:
                con.execute(
                    "UPDATE scene SET display_name = ?, named = ? WHERE scope = ?",
                    (display_name, int(named), scope),
                )

    def touch_blob(self, name: str) -> dict | None:
        """Mark a file as used now; its row, or None when it has none."""
        with self._db(write=True) as con:
            cur = con.execute("UPDATE blob SET last_used = ? WHERE name = ?", (time.time(), name))
            if not cur.rowcount:
                return None
            row = con.execute(
                "SELECT kind, size, key_id, sha256 FROM blob WHERE name = ?", (name,)
            ).fetchone()
        return {"kind": row[0], "size": row[1], "key_id": row[2], "sha256": row[3]}

    def record_blob(self, name: str, kind: str, size: int, *, key_id=None, sha256=None) -> None:
        now = time.time()
        # No UPSERT (INSERT ... ON CONFLICT): it needs SQLite 3.24, and Python
        # builds on older systems link 3.22 or less.
        with self._db(write=True) as con:
            cur = con.execute(
                "UPDATE blob SET kind = ?, size = ?, key_id = ?, sha256 = ?, last_used = ?"
                " WHERE name = ?",
                (kind, size, key_id, sha256, now, name),
            )
            if not cur.rowcount:
                con.execute(
                    "INSERT INTO blob (name, kind, size, key_id, sha256, created, last_used)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (name, kind, size, key_id, sha256, now, now),
                )

    def link(self, scope: str, name: str) -> None:
        with self._db(write=True) as con:
            con.execute(
                "INSERT OR IGNORE INTO scene_blob (scope, name) VALUES (?, ?)", (scope, name)
            )

    def forget(self, name: str, kind: str) -> int:
        """Delete a file and its rows; the caller holds the name's lock.
        Returns the bytes freed."""
        path = self.path_of(name, kind)
        try:
            size = path.stat().st_size
            path.unlink()
        except FileNotFoundError:
            size = 0
        with self._db(write=True) as con:
            con.execute("DELETE FROM blob WHERE name = ?", (name,))
            con.execute("DELETE FROM scene_blob WHERE name = ?", (name,))
        return size

    def _try_forget(
        self, name: str, kind: str, still: Callable[[sqlite3.Connection], bool]
    ) -> int | None:
        """Delete a file when its lock is free and ``still(con)`` holds under
        the lock (a session may have used it since it was picked). Returns
        the bytes freed, or None when the file stays."""
        lock = self.lock(name)
        if not lock.acquire(wait=False):
            return None
        try:
            with self._db() as con:
                if not still(con):
                    return None
            return self.forget(name, kind)
        finally:
            lock.release(remove=True)

    # ----- scenes ----------------------------------------------------------- #
    def scenes(self) -> list[dict]:
        with self._db() as con:
            rows = con.execute(
                "SELECT s.scope, s.display_name, s.encrypted, s.pinned, s.last_used,"
                " COUNT(b.name), COALESCE(SUM(b.size), 0)"
                " FROM scene s LEFT JOIN scene_blob sb ON sb.scope = s.scope"
                " LEFT JOIN blob b ON b.name = sb.name"
                " GROUP BY s.scope ORDER BY s.last_used DESC"
            ).fetchall()
        return [
            {
                "scene": r[1],
                "scope": r[0],
                "encrypted": bool(r[2]),
                "pinned": bool(r[3]),
                "last_used": _utc(r[4]),
                "files": r[5],
                "size": r[6],
            }
            for r in rows
        ]

    @staticmethod
    def _match(con, scene: str) -> list[str]:
        rows = con.execute(
            "SELECT scope FROM scene WHERE display_name = ? OR scope = ?", (scene, scene)
        ).fetchall()
        return [r[0] for r in rows]

    def set_pinned(self, scene: str, pinned: bool) -> list[dict]:
        with self._db(write=True) as con:
            scopes = self._match(con, scene)
            if not scopes:
                raise LookupError(f"no cached scene is called {scene!r}: ls shows what is cached")
            for scope in scopes:
                con.execute("UPDATE scene SET pinned = ? WHERE scope = ?", (int(pinned), scope))
        return [s for s in self.scenes() if s["scope"] in scopes]

    # ----- clearing --------------------------------------------------------- #
    def clear(self, scene: str | None = None) -> dict:
        """Delete one scene's files, or every file when ``scene`` is None.
        Files another scene uses too stay; files being downloaded right now
        stay."""
        with self.batch():
            return self._clear(scene)

    def _clear(self, scene: str | None) -> dict:
        freed = files = 0
        with self._db() as con:
            if scene is None:
                scopes = [r[0] for r in con.execute("SELECT scope FROM scene")]
                names = con.execute("SELECT name, kind FROM blob").fetchall()
            else:
                scopes = self._match(con, scene)
                if not scopes:
                    raise LookupError(
                        f"no cached scene is called {scene!r}: ls shows what is cached"
                    )
                marks = ",".join("?" * len(scopes))
                names = con.execute(
                    "SELECT b.name, b.kind FROM blob b JOIN scene_blob sb ON sb.name = b.name"
                    f" WHERE sb.scope IN ({marks}) AND NOT EXISTS (SELECT 1 FROM scene_blob o"
                    f" WHERE o.name = b.name AND o.scope NOT IN ({marks}))",
                    (*scopes, *scopes),
                ).fetchall()
        for name, kind in names:
            lock = self.lock(name)
            if not lock.acquire(wait=False):
                continue
            try:
                freed += self.forget(name, kind)
                files += 1
            finally:
                lock.release(remove=True)
        with self._db(write=True) as con:
            for scope in scopes:
                con.execute("DELETE FROM scene_blob WHERE scope = ?", (scope,))
                con.execute("DELETE FROM scene WHERE scope = ?", (scope,))
        if scene is None:
            for part in self.tmp.glob("*.part"):
                if not self._part_is_live(part):
                    with contextlib.suppress(OSError):
                        part.unlink()
        return {"scenes": len(scopes), "files": files, "bytes": freed}

    def clear_legacy(self) -> dict:
        """Delete what clients before 0.6 left: plain files named by their
        digest at the top of a ``cache_dir``, and ``we_render_*`` temporary
        directories unused for an hour."""
        return _clear_legacy_without_cache(self.root)

    def legacy_files(self) -> int:
        try:
            return sum(1 for p in self.root.iterdir() if dac1.is_digest(p.name))
        except OSError:
            return 0

    # ----- retention -------------------------------------------------------- #
    def maybe_cleanup(self, *, background: bool = True) -> bool:
        """Run :meth:`cleanup` when the last one is a day old; one process
        does it. Returns whether this call started one."""
        now = time.time()
        try:
            with self._db(write=True) as con:
                last = float(self._meta(con, "last_cleanup") or 0)
                if now - last < _CLEANUP_EVERY:
                    return False
                con.execute(
                    "INSERT OR REPLACE INTO meta (k, v) VALUES ('last_cleanup', ?)", (repr(now),)
                )
        except (sqlite3.Error, ValueError) as exc:
            log.warning("render cache cleanup skipped: %s", exc)
            return False

        def run():
            try:
                self.cleanup(now=now)
            except Exception as exc:  # noqa: BLE001 - never break a session for this
                log.warning("render cache cleanup failed: %s", exc)

        if background:
            threading.Thread(target=run, name="render-cache-cleanup", daemon=True).start()
        else:
            run()
        return True

    def cleanup(self, now: float | None = None) -> dict:
        """Apply the retention settings now. Files used within the last hour
        always stay, and so do files being downloaded."""
        with self.batch():
            return self._cleanup(time.time() if now is None else now)

    def _cleanup(self, now: float) -> dict:
        s = self.settings()
        freed = files = 0
        unpinned = (
            "name NOT IN (SELECT sb.name FROM scene_blob sb JOIN scene sc"
            " ON sc.scope = sb.scope WHERE sc.pinned = 1)"
        )

        def unused_since(cutoff):
            def still(con, name):
                row = con.execute("SELECT last_used FROM blob WHERE name = ?", (name,)).fetchone()
                return row is not None and row[0] < cutoff

            return still

        # Age.
        if s["max_age_days"] > 0:
            cutoff = now - max(s["max_age_days"] * _DAY, _IN_USE)
            with self._db() as con:
                old = con.execute(
                    f"SELECT name, kind FROM blob WHERE last_used < ? AND {unpinned}", (cutoff,)
                ).fetchall()
            still = unused_since(cutoff)
            for name, kind in old:
                got = self._try_forget(name, kind, lambda con, n=name: still(con, n))
                if got is not None:
                    freed, files = freed + got, files + 1
        # Size.
        if s["max_size_gb"]:
            limit = s["max_size_gb"] * 1e9
            cutoff = now - _IN_USE
            with self._db() as con:
                total = con.execute("SELECT COALESCE(SUM(size), 0) FROM blob").fetchone()[0]
                lru = con.execute(
                    f"SELECT name, kind, size FROM blob WHERE last_used < ? AND {unpinned}"
                    " ORDER BY last_used ASC",
                    (cutoff,),
                ).fetchall()
            still = unused_since(cutoff)
            for name, kind, size in lru:
                if total <= limit:
                    break
                got = self._try_forget(name, kind, lambda con, n=name: still(con, n))
                if got is not None:
                    total, freed, files = total - size, freed + got, files + 1
        # Rows whose file is gone, files no row knows, scenes without files.
        with self._db() as con:
            rows = con.execute("SELECT name, kind FROM blob").fetchall()
        known = {name for name, _ in rows}
        for name, kind in rows:
            path = self.path_of(name, kind)
            if not path.exists():
                self._try_forget(name, kind, lambda con, p=path: not p.exists())
        for top in (self.blobs, self.pages):
            for path in top.glob("*/*"):
                try:
                    stale = path.name not in known and now - path.stat().st_mtime > _STALE_TEMP
                except OSError:
                    continue
                if not stale:
                    continue
                lock = self.lock(path.name)
                if not lock.acquire(wait=False):
                    continue
                try:
                    with self._db() as con:
                        orphan = (
                            con.execute(
                                "SELECT 1 FROM blob WHERE name = ?", (path.name,)
                            ).fetchone()
                            is None
                        )
                    if orphan:
                        with contextlib.suppress(OSError):
                            size = path.stat().st_size
                            path.unlink()
                            freed, files = freed + size, files + 1
                finally:
                    lock.release(remove=True)
        with self._db(write=True) as con:
            con.execute("DELETE FROM scene_blob WHERE name NOT IN (SELECT name FROM blob)")
            con.execute(
                "DELETE FROM scene WHERE pinned = 0 AND last_used < ?"
                " AND scope NOT IN (SELECT scope FROM scene_blob)",
                (now - _IN_USE,),
            )
        # Leftover downloads and lock files.
        for part in self.tmp.glob("*.part"):
            if not self._part_is_live(part, age=_STALE_TEMP, now=now):
                with contextlib.suppress(OSError):
                    part.unlink()
        if os.name != "nt":
            for lock_file in self.locks.glob("*.lock"):
                if lock_file.name.startswith("."):
                    continue  # the index's own lock
                lock = self.lock(lock_file.name[: -len(".lock")])
                if lock.acquire(wait=False):
                    lock.release(remove=True)
        dirs, _ = remove_temp_dirs(age=_STALE_TEMP, now=now)
        return {"files": files, "bytes": freed, "temp_dirs": dirs}

    def _part_is_live(self, part: Path, *, age: float | None = None, now: float = 0.0) -> bool:
        """Whether a download may still own ``part``: this process wrote it,
        or its process still runs (on a local disk, where the process id is
        this machine's) and, when ``age`` is given, it is younger than that."""
        try:
            pid = int(part.name.split(".")[-3])
        except (IndexError, ValueError):
            pid = -1
        if pid == os.getpid():
            return True
        if not self.network_fs and os.name != "nt" and not _pid_alive(pid):
            return False
        if age is None:
            return True
        try:
            return (now or time.time()) - part.stat().st_mtime <= age
        except OSError:
            return False

    def drop_other_keys(self, scope: str, key_id: str) -> int:
        """A scene's key changed (a new epoch, a key restored after a
        revocation): files under its old keys can never be decrypted again.
        Delete those not used for a day (a session that started before the
        change may still read them)."""
        cutoff = time.time() - _OLD_KEY_GRACE
        with self._db() as con:
            old = con.execute(
                "SELECT b.name FROM blob b JOIN scene_blob sb ON sb.name = b.name"
                " WHERE sb.scope = ? AND b.kind = 'dac1' AND b.key_id != ? AND b.last_used < ?",
                (scope, key_id, cutoff),
            ).fetchall()
        freed = 0
        for (name,) in old:

            def still(con, name=name):
                row = con.execute("SELECT last_used FROM blob WHERE name = ?", (name,)).fetchone()
                return row is not None and row[0] < cutoff

            freed += self._try_forget(name, "dac1", still) or 0
        return freed


def _check_age(value, what: str) -> float:
    try:
        age = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number of days (0 keeps files forever)") from None
    if not age >= 0 or age == float("inf") or isinstance(value, bool):
        raise ValueError(f"{what} must be 0 or more days (0 keeps files forever)")
    return age


def _check_size(value, what: str) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none")):
        return None
    try:
        size = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number of gigabytes, or none") from None
    if not size > 0 or size == float("inf") or isinstance(value, bool):
        raise ValueError(f"{what} must be more than 0 gigabytes, or none")
    return size


def remove_temp_dirs(*, age: float, now: float | None = None) -> tuple[int, int]:
    """Delete this user's ``we_render_*`` directories in the temporary
    directory that nothing touched for ``age`` seconds. Before 0.6 the
    renderer kept plain assets there and deleted them only on close, so a
    killed process left them behind. Returns (directories, files) removed."""
    now = time.time() if now is None else now
    removed = files = 0
    try:
        candidates = sorted(Path(tempfile.gettempdir()).glob("we_render_*"))
    except OSError:
        return 0, 0
    for d in candidates:
        try:
            st = d.lstat()
            if not d.is_dir() or d.is_symlink():
                continue
            if os.name != "nt" and st.st_uid != os.getuid():
                continue
            newest, count = st.st_mtime, 0
            for sub, _dirs, names in os.walk(d):
                for n in names:
                    with contextlib.suppress(OSError):
                        newest = max(newest, os.lstat(os.path.join(sub, n)).st_mtime)
                        count += 1
            if now - newest < age:
                continue
        except OSError:
            continue
        shutil.rmtree(d, ignore_errors=True)
        if not d.exists():
            removed += 1
            files += count
    return removed, files


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #
@dataclass
class SceneKey:
    """The platform's key for one scene: this account's own, or one that all
    the accounts that may read the scene share. In memory only."""

    scope: str
    key_id: bytes
    key: bytes = field(repr=False)


def _refusal(exc: RenderFetchError, what: str) -> RendererError:
    if exc.status == 403 and exc.code == "asset_key_revoked":
        return RendererError(
            f"the platform refused {what}: {exc.message}",
            code="asset_key_revoked",
            fixes=[
                "ask the scene's owner whether you still have access",
                "or connect with images=False for a state-only session",
            ],
        )
    fixes = list(exc.fixes or []) or ["try again later"]
    if exc.status == 429:
        return RendererError(
            f"the platform refused {what}: {exc.message}",
            code=exc.code or "rate_limited",
            fixes=fixes,
        )
    return RendererError(
        f"the platform refused {what} (HTTP {exc.status}): {exc.message}",
        code=exc.code or "asset_unavailable",
        fixes=fixes,
    )


def fetch_scene_key(fetch) -> SceneKey:
    """Ask the platform for this session's scene key (``GET /render/__key``;
    the platform checks read access again on every request)."""
    try:
        with fetch("/__key") as response:
            raw = response.read(_KEY_REPLY_MAX + 1)
    except FileNotFoundError:
        raise RendererError(
            "the platform sends this scene's assets encrypted but sent no key for them "
            "(no /render/__key)",
            code="asset_key_unavailable",
            fixes=["try again later", "or connect with images=False for a state-only session"],
        ) from None
    except RenderFetchError as exc:
        raise _refusal(exc, "this scene's asset key") from None
    try:
        if len(raw) > _KEY_REPLY_MAX:
            raise ValueError("too long")
        reply = json.loads(raw)
        if not isinstance(reply, dict):
            raise ValueError("not an object")
        if reply.get("v") != 1 or reply.get("alg") != "A256GCM":
            raise ValueError(f"version {reply.get('v')!r}, algorithm {reply.get('alg')!r}")
        if reply.get("format", "dac1") != "dac1":
            raise ValueError(f"format {reply.get('format')!r}")
        key = base64.b64decode(reply["key"], validate=True)
        kid = bytes.fromhex(reply["key_id"])
        scope = reply["scope"]
        if len(key) != dac1.KEY_LEN or len(kid) != 16:
            raise ValueError("wrong key or key id length")
        if not isinstance(scope, str) or not scope or len(scope) > 200:
            raise ValueError("no scene in the reply")
        if dac1.key_id(key) != kid:
            raise ValueError("the key id does not match the key")
    except (ValueError, KeyError, TypeError) as exc:
        raise RendererError(
            f"the platform's asset key reply is not readable: {exc}",
            code="asset_key_unavailable",
            fixes=["pip install -U 'world-engine-client[render]'"],
        ) from None
    return SceneKey(scope=scope, key_id=kid, key=key)


# --------------------------------------------------------------------------- #
# One session's assets
# --------------------------------------------------------------------------- #
class _Mismatch(Exception):
    pass


class _PlainCheck:
    """Checks a plain download against its digest; passes the bytes on."""

    def __init__(self, digest: str):
        self._digest, self._sha = digest, hashlib.sha256()

    def update(self, data: bytes) -> list[bytes]:
        self._sha.update(data)
        return [data]

    def finish(self) -> None:
        if self._sha.hexdigest() != self._digest:
            raise _Mismatch("does not match its digest")


def _check_plain_file(path: Path, digest: str) -> int:
    sha, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            sha.update(data)
            size += len(data)
    if sha.hexdigest() != digest:
        raise _Mismatch("does not match its digest")
    return size


def _iter_plain(path: Path, digest: str) -> Iterator[bytes]:
    check = _PlainCheck(digest)
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            yield from check.update(data)
    check.finish()


@dataclass
class _Entry:
    url: str
    digest: str
    wire: str  # how it arrives: dac1 | plain
    kept: str  # where it is kept: disk | memory
    name: str = ""
    length: int = 0


def _content_range_start(value: str | None) -> int | None:
    m = re.fullmatch(r"bytes (\d+)-\d+/(?:\d+|\*)", (value or "").strip())
    return int(m.group(1)) if m else None


def _header(response, name: str) -> str | None:
    get = getattr(response, "getheader", None)
    return get(name) if get else None


def _int_header(response, name: str) -> int | None:
    value = _header(response, name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _transport_errors() -> tuple[type[BaseException], ...]:
    import http.client

    return (OSError, TimeoutError, http.client.HTTPException)


class _StallWatch:
    """While waiting for another download of a name: True once that download
    has not grown for _STALL_SECONDS (a process stopped while holding it)."""

    def __init__(self, cache: RenderCache, name: str):
        self._cache, self._name = cache, name
        self._size, self._since = -1, time.monotonic()

    def __call__(self) -> bool:
        try:
            size = sum(p.stat().st_size for p in self._cache.parts(self._name))
        except OSError:
            size = self._size
        if size != self._size:
            self._size, self._since = size, time.monotonic()
        return time.monotonic() - self._since > _STALL_SECONDS


class SceneAssets:
    """The scene assets of one rendering session.

    ``prepare()`` fetches the scene key (encrypted sessions), then makes sure
    every asset in ``digests`` is cached and verified, downloading what is
    missing or damaged. ``open(url)`` and ``read(url)`` give the verified
    plaintext, in memory only.

    How assets are kept follows from the connect bundle:

    * ``encrypted``: the platform sends scene assets (``/__asset/*``) as
      DAC1. They are stored as received and decrypted with the session's key.
      The robot meshes (``/__robot/*``, public MuJoCo Menagerie geometry)
      arrive plain and are stored plain.
    * ``plain``: your own local ``world-engine serve`` sends your scene as it
      is; files are stored plain and checked against their digests.
    * ``memory``: a platform session whose assets arrive unencrypted (a
      platform from before its encryption): nothing is written to disk.

    ``fetch(path)`` opens ``<server>/render<path>``; ``fetch(path, headers)``
    adds request headers (a ``Range`` to resume a download)."""

    def __init__(
        self,
        fetch,
        digests: dict[str, str],
        *,
        encrypted: bool,
        platform: bool,
        cache: RenderCache | None = None,
        cache_dir: str | os.PathLike | None = None,
        scene_name: str | None = None,
        scene_id: str | None = None,
        origin: str = "",
    ):
        self._fetch = fetch
        self._digests = dict(digests)
        self.mode = "encrypted" if encrypted else ("memory" if platform else "plain")
        self._origin = origin
        self._scene_name = scene_name
        self._scene_id = scene_id
        self._key: SceneKey | None = None
        self._entries: dict[str, _Entry] = {}
        self._memory: dict[str, bytes] = {}
        self.scope: str | None = None
        self.downloaded: list[str] = []  # the URLs this session downloaded
        self.cache = cache
        if self.cache is None:
            try:
                self.cache = RenderCache(cache_dir)
            except CacheUnavailable as exc:
                warnings.warn(
                    f"{exc}. Scene assets are kept in memory for this session and downloaded "
                    "again next time; set DIGIARENA_CACHE_DIR to a writable directory on a "
                    "local disk",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self.mode == "memory":
            _warn_once(
                "memory",
                "this platform sent scene assets unencrypted: they are kept in memory only, "
                "not cached, and downloaded again for every session",
            )

    # ----- setup ------------------------------------------------------------ #
    @property
    def urls(self) -> list[str]:
        return sorted(self._digests)

    @property
    def key(self) -> SceneKey | None:
        return self._key

    def prepare(self, *, verify_hits: bool = True, cleanup: bool = True) -> None:
        """Get the key, then download or verify every asset. ``verify_hits``
        False checks only the size and header of a cached file (a caller
        that decrypts everything before use anyway). ``cleanup`` then applies
        the retention settings in the background, at most once a day; the
        files of this session were just used, so they stay."""
        for url, digest in self._digests.items():
            if not isinstance(url, str) or not _ASSET_URL.fullmatch(url) or "/../" in url + "/":
                raise RendererError(
                    f"the server named an unusable asset URL {url!r}", code="asset_mismatch"
                )
            if not dac1.is_digest(digest):
                raise RendererError(
                    f"the server's digest for {url} is not 64 lowercase hex characters",
                    code="asset_mismatch",
                )
        if self.mode == "encrypted":
            self._key = fetch_scene_key(self._fetch)
            self.scope = self._key.scope
        else:
            self.scope = f"local:{self._origin}/{self._scene_id or 'scene'}"
        cache = self.cache
        with cache.batch() if cache is not None else contextlib.nullcontext():
            if cache is not None and self.mode != "memory":
                if cache.legacy_files():
                    _warn_once(
                        f"legacy:{cache.root}",
                        f"{cache.root} holds plain scene files from a world-engine-client "
                        "before 0.6; they are no longer read. "
                        f"python -m world_engine_client.cache --dir {cache.root} clear "
                        "--legacy deletes them",
                    )
                cache.touch_scene(
                    self.scope,
                    self._scene_name or self._scene_id or self.scope,
                    named=bool(self._scene_name),
                    encrypted=self.mode == "encrypted",
                    key_id=self._key.key_id.hex() if self._key else None,
                )
            for url in self.urls:
                entry = self._entry(url)
                self._ensure(entry, verify_hits)
                self._entries[url] = entry
                if entry.kept == "disk":
                    cache.link(self.scope, entry.name)
            if self._key is not None and cache is not None:
                cache.drop_other_keys(self.scope, self._key.key_id.hex())
        if cleanup and cache is not None:
            # Only now: a scene opened after more than max_age_days unused was
            # deleted by its own connect's cleanup and downloaded again.
            cache.maybe_cleanup()

    def _entry(self, url: str) -> _Entry:
        digest = self._digests[url]
        wire = "dac1" if self.mode == "encrypted" and url.startswith("/__asset/") else "plain"
        if self.cache is None or self.mode == "memory":
            return _Entry(url, digest, wire, "memory")
        name = dac1.file_name(self._key.key, digest) if wire == "dac1" else digest
        return _Entry(url, digest, wire, "disk", name)

    def _verify(self, entry: _Entry, path: Path) -> int:
        """The plaintext length of a cached file, after checking all of it."""
        if entry.wire == "dac1":
            return sum(len(c) for c in dac1.iter_file(self._key.key, entry.digest, path))
        return _check_plain_file(path, entry.digest)

    def _length(self, entry: _Entry, path: Path) -> int:
        if entry.wire == "dac1":
            header = dac1.read_header(path)
            if path.stat().st_size != header.cipher_length:
                raise dac1.Dac1Error("truncated or padded")
            if header.key_id != self._key.key_id:
                raise dac1.Dac1Error("encrypted for another key")
            return header.length
        return path.stat().st_size

    def _ensure(self, entry: _Entry, verify_hits: bool) -> None:
        if entry.kept == "memory":
            self._memory[entry.url] = self._download_memory(entry)
            entry.length = len(self._memory[entry.url])
            return
        cache = self.cache
        path = cache.blob_path(entry.name)
        lock = cache.lock(entry.name)
        # Looked up under the name's lock, which also waits for a process
        # downloading the file. Cleanup deletes a file only under this lock,
        # after reading its last_used again: once touched here, it stays for
        # this session.
        held = lock.acquire(give_up=_StallWatch(cache, entry.name))
        try:
            if cache.touch_blob(entry.name) is None or not path.exists():
                self._fill(entry, path, held)
                return
        finally:
            if held:
                lock.release(remove=True)
        # A file in place never changes, so it is checked outside the lock,
        # while other processes check their copies of it.
        try:
            entry.length = (self._verify if verify_hits else self._length)(entry, path)
            return
        except (dac1.Dac1Error, _Mismatch, OSError) as exc:
            log.info("cached %s is damaged (%s): downloading it again", entry.url, exc)
        held = lock.acquire(give_up=_StallWatch(cache, entry.name))
        try:
            self._fill(entry, path, held)
        finally:
            if held:
                lock.release(remove=True)

    def _fill(self, entry: _Entry, path: Path, held: bool) -> None:
        """With the name's lock (``held``; False when its holder stalled): keep
        the file in place if it verifies (another process stored it, or died
        between storing and recording it), else download it."""
        cache = self.cache
        if path.exists():
            try:
                entry.length = self._verify(entry, path)
                self._record(entry, path)
                return
            except (dac1.Dac1Error, _Mismatch, OSError) as exc:
                log.info("cached %s is damaged (%s): downloading it again", entry.url, exc)
                if held:
                    cache.forget(entry.name, "blob")
        if held:
            cache.remove_parts(entry.name)
        self._download(entry)

    def _record(self, entry: _Entry, path: Path) -> None:
        self.cache.record_blob(
            entry.name,
            entry.wire,
            path.stat().st_size,
            key_id=self._key.key_id.hex() if entry.wire == "dac1" else None,
        )

    def _checker(self, entry: _Entry):
        if entry.wire == "dac1":
            return dac1.Decryptor(self._key.key, entry.digest)
        return _PlainCheck(entry.digest)

    def _get(self, url: str, headers: dict | None):
        return self._fetch(url, headers) if headers else self._fetch(url)

    def _stream(
        self,
        entry: _Entry,
        write: Callable[[bytes], None] | None,
        plain: Callable[[bytes], None] | None,
        restart: Callable[[], None],
    ) -> int:
        """Download ``entry.url``: the bytes as received go to ``write``, the
        verified plaintext to ``plain``. A dropped connection resumes with a
        Range; ``restart()`` empties both when the server answers a Range
        with the whole file. Returns the bytes received."""
        check = self._checker(entry)
        written = attempts = 0
        while True:
            headers = {"Range": f"bytes={written}-"} if written else None
            try:
                with self._get(entry.url, headers) as response:
                    status = getattr(response, "status", 200)
                    start = _content_range_start(_header(response, "Content-Range"))
                    if written and (status != 206 or start != written):
                        restart()
                        check, written = self._checker(entry), 0
                    if entry.wire == "dac1" and written == 0:
                        first = response.read(dac1.HEADER_LEN)
                        if first[:4] != dac1.MAGIC:
                            raise RendererError(
                                f"the platform sent {entry.url} unencrypted, not as DAC1",
                                code="asset_mismatch",
                            )
                        pieces = [first]
                    else:
                        pieces = []
                    expected = _int_header(response, "Content-Length")
                    got = 0
                    rest = iter(lambda: response.read(1 << 20), b"")
                    for data in itertools.chain(pieces, rest):
                        if entry.wire == "plain" and written == 0 and data[:4] == dac1.MAGIC:
                            raise RendererError(
                                f"the server sent {entry.url} encrypted; this client expects "
                                "it plain",
                                code="client_upgrade_required",
                                fixes=["pip install -U 'world-engine-client[render]'"],
                            )
                        out = check.update(data)
                        if write is not None:
                            write(data)
                        if plain is not None:
                            for chunk in out:
                                plain(chunk)
                        written += len(data)
                        got += len(data)
                    # http.client's read(n) ends quietly on a connection that
                    # closed early: a body shorter than announced broke off.
                    if expected is not None and got < expected:
                        raise ConnectionError(f"the body ended after {got} of {expected} bytes")
                check.finish()
                return written
            except (dac1.Dac1Error, _Mismatch) as exc:
                raise RendererError(
                    f"asset {entry.url} does not match its digest: {exc}", code="asset_mismatch"
                ) from None
            except FileNotFoundError:
                raise RendererError(
                    f"the server does not have the scene asset {entry.url}",
                    code="renderer_unavailable",
                ) from None
            except RenderFetchError as exc:
                if exc.status not in (502, 503, 504) or attempts >= _RETRIES:
                    raise _refusal(exc, f"the scene asset {entry.url}") from None
            except _transport_errors() as exc:
                if attempts >= _RETRIES:
                    raise RendererError(
                        f"downloading {entry.url} failed: {exc}",
                        code="renderer_unavailable",
                        fixes=["check the network connection and connect again"],
                    ) from None
                log.info(
                    "download of %s stopped at %d bytes (%s): resuming", entry.url, written, exc
                )
            attempts += 1
            time.sleep(min(0.25 * 2**attempts, 4.0))

    def _download(self, entry: _Entry) -> None:
        cache = self.cache
        part, fd = cache.new_part(entry.name)
        try:
            with os.fdopen(fd, "wb") as out:

                def restart():
                    out.seek(0)
                    out.truncate()

                self._stream(entry, out.write, None, restart)
            final = cache.install(part, entry.name, "blob", lambda p: self._verify(entry, p))
        except BaseException:
            with contextlib.suppress(OSError):
                part.unlink()
            raise
        entry.length = self._length(entry, final)
        self._record(entry, final)
        self.downloaded.append(entry.url)

    def _download_memory(self, entry: _Entry) -> bytes:
        buf = bytearray()
        self._stream(entry, None, buf.extend, buf.clear)
        self.downloaded.append(entry.url)
        return bytes(buf)

    # ----- use -------------------------------------------------------------- #
    def length(self, url: str) -> int:
        """The plaintext size of one of this session's assets."""
        return self._entries[url].length

    def open(self, url: str) -> Iterator[bytes]:
        """The verified plaintext of ``url``, piece by piece; each piece only
        after its check passed. KeyError for a URL that is not one of this
        session's assets."""
        entry = self._entries[url]
        if entry.kept == "memory":
            data = self._memory[url]
            return iter([data[i : i + (1 << 22)] for i in range(0, len(data), 1 << 22)] or [b""])
        path = self.cache.blob_path(entry.name)
        if entry.wire == "dac1":
            return dac1.iter_file(self._key.key, entry.digest, path)
        return _iter_plain(path, entry.digest)

    def read(self, url: str) -> bytes:
        """The whole verified plaintext of ``url``, in memory."""
        return b"".join(self.open(url))

    def page_file(self, path: str) -> bytes:
        """A file of the capture page (index.html and its scripts). Scripts
        whose name carries Vite's content hash are kept, per server: one name,
        one content. Everything else is fetched every time."""
        cache = self.cache
        if cache is None or not _HASHED_PAGE_FILE.fullmatch(path):
            return self._read_upstream(path)
        name = hashlib.sha256(f"{self._origin}\0{path}".encode()).hexdigest()[:40]
        final = cache.page_path(name)

        def cached() -> bytes | None:
            row = cache.touch_blob(name)
            if row is None or row["kind"] != "page":
                return None
            try:
                data = final.read_bytes()
            except OSError:
                return None
            return data if hashlib.sha256(data).hexdigest() == row["sha256"] else None

        data = cached()
        if data is not None:
            return data
        lock = cache.lock(name)
        if not lock.acquire(give_up=_StallWatch(cache, name)):
            return self._read_upstream(path)
        try:
            data = cached()  # stored by someone else while we waited
            if data is not None:
                return data
            data = self._read_upstream(path)
            sha = hashlib.sha256(data).hexdigest()
            part, fd = cache.new_part(name)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(data)

                def same(p: Path) -> None:
                    if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
                        raise _Mismatch("another copy")

                cache.install(part, name, "page", same)
            except BaseException:
                with contextlib.suppress(OSError):
                    part.unlink()
                raise
            cache.record_blob(name, "page", len(data), sha256=sha)
            return data
        finally:
            lock.release(remove=True)

    def _read_upstream(self, path: str) -> bytes:
        with self._fetch(path) as response:
            return response.read()

    def close(self) -> None:
        """Forget the key and whatever is kept in memory."""
        self._key = None
        self._memory.clear()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _existing(cache_dir) -> RenderCache | None:
    """The cache at ``cache_dir`` when there is one: what only reads or
    removes leaves a machine without a cache as it is."""
    if not (directory(cache_dir) / "index.sqlite3").is_file():
        return None
    return RenderCache(cache_dir)


def _no_scene(scene: str) -> LookupError:
    return LookupError(f"no cached scene is called {scene!r}: ls shows what is cached")


def list_scenes(cache_dir: str | os.PathLike | None = None) -> list[dict]:
    """The cached scenes, most recently used first:
    ``[{scene, scope, encrypted, pinned, last_used, files, size}]``. ``scene``
    is the name the scene was connected with (``owner/slug`` through the
    DigiArena SDK), ``scope`` the platform's scene id; ``size`` is in bytes."""
    cache = _existing(cache_dir)
    return cache.scenes() if cache is not None else []


def pin(scene: str, cache_dir: str | os.PathLike | None = None) -> list[dict]:
    """Keep a cached scene however long it goes unused (by name or scope).
    Raises LookupError for a scene that is not cached."""
    cache = _existing(cache_dir)
    if cache is None:
        raise _no_scene(scene)
    return cache.set_pinned(scene, True)


def unpin(scene: str, cache_dir: str | os.PathLike | None = None) -> list[dict]:
    """Let a pinned scene age out again."""
    cache = _existing(cache_dir)
    if cache is None:
        raise _no_scene(scene)
    return cache.set_pinned(scene, False)


def clear(
    scene: str | None = None,
    *,
    legacy: bool = False,
    cache_dir: str | os.PathLike | None = None,
) -> dict:
    """Delete a scene's cached files, or every cached file when ``scene`` is
    None. ``legacy=True`` also deletes what clients before 0.6 left: plain
    files in a ``cache_dir`` and ``we_render_*`` temporary directories unused
    for an hour. ``clear(legacy=True)`` without a scene deletes only those.
    Returns what was removed."""
    cache = _existing(cache_dir)
    out: dict = {}
    if scene is not None or not legacy:
        if cache is None and scene is not None:
            raise _no_scene(scene)
        out = cache.clear(scene) if cache is not None else {"scenes": 0, "files": 0, "bytes": 0}
    if legacy:
        root = directory(cache_dir)
        out["legacy"] = (
            cache.clear_legacy() if cache is not None else _clear_legacy_without_cache(root)
        )
    return out


def _clear_legacy_without_cache(root: Path) -> dict:
    """clear(legacy=True) for an old ``cache_dir`` this version never used."""
    freed = files = 0
    if root.is_dir():
        for path in root.iterdir():
            stem = path.name[: -len(".part")] if path.name.endswith(".part") else path.name
            if dac1.is_digest(stem) and path.is_file() and not path.is_symlink():
                with contextlib.suppress(OSError):
                    size = path.stat().st_size
                    path.unlink()
                    freed += size
                    files += 1
    # Spare what an old client running on this machine may still be loading.
    dirs, _ = remove_temp_dirs(age=_IN_USE)
    return {"files": files, "bytes": freed, "temp_dirs": dirs}


def configure(
    *,
    max_age_days=_UNSET,
    max_size_gb=_UNSET,
    cache_dir: str | os.PathLike | None = None,
) -> dict:
    """Change the retention settings (arguments left out stay as they are)
    and return the settings in force. ``max_age_days=0`` keeps files however
    old; ``max_size_gb=None`` removes the size limit. Environment variables,
    when set, still win (``from_environment`` lists them)."""
    return RenderCache(cache_dir).configure(max_age_days=max_age_days, max_size_gb=max_size_gb)


def settings(cache_dir: str | os.PathLike | None = None) -> dict:
    """The settings in force and the cache directory."""
    cache = _existing(cache_dir)
    if cache is not None:
        return cache.settings()
    return _Defaults(directory(cache_dir)).settings()


def cleanup(cache_dir: str | os.PathLike | None = None) -> dict:
    """Apply the retention settings now (renderers do it at most once a day)."""
    cache = _existing(cache_dir)
    return cache.cleanup() if cache is not None else {"files": 0, "bytes": 0, "temp_dirs": 0}


class _Defaults(RenderCache):
    """The settings of a cache that does not exist yet: nothing stored."""

    def __init__(self, root: Path):  # no directories, no index
        self.root = root

    @contextlib.contextmanager
    def _db(self, write: bool = False):
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            yield con
        finally:
            con.close()


def __getattr__(name: str):
    # world_engine_client.cache.list(), like digiarena.cache.list(), without a
    # module-level ``list`` that would hide the builtin inside this module.
    if name == "list":
        return list_scenes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _jsonable(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(type(obj).__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The cache commands, for this module's CLI and the DigiArena SDK's."""
    parser.add_argument(
        "--dir",
        help="the cache directory (default: DIGIARENA_CACHE_DIR, else the user cache directory)",
    )
    sub = parser.add_subparsers(dest="cache_cmd", required=True)
    sub.add_parser("ls", help="list the cached scenes")
    p_pin = sub.add_parser("pin", help="keep a scene however long it goes unused")
    p_pin.add_argument("scene")
    p_unpin = sub.add_parser("unpin", help="let a pinned scene age out again")
    p_unpin.add_argument("scene")
    p_clear = sub.add_parser("clear", help="delete a scene's cached files, or all of them")
    p_clear.add_argument("scene", nargs="?")
    p_clear.add_argument("--all", action="store_true", help="every cached file")
    p_clear.add_argument(
        "--legacy",
        action="store_true",
        help="plain files that clients before 0.6 left in --dir, and their we_render_* "
        "temporary directories unused for an hour",
    )
    p_config = sub.add_parser("config", help="show or change the retention settings")
    p_config.add_argument("--max-age-days", help="days unused before a file goes; 0 keeps all")
    p_config.add_argument("--max-size-gb", help="a size limit in GB, or none")


def run(args: argparse.Namespace, out=None, err=None) -> int:
    """Run a parsed cache command; returns the exit code."""
    out = out or sys.stdout
    err = err or sys.stderr

    def show(obj):
        print(json.dumps(obj, indent=2, default=_jsonable), file=out)

    try:
        cmd = args.cache_cmd
        if cmd == "ls":
            show(list_scenes(args.dir))
        elif cmd in ("pin", "unpin"):
            show((pin if cmd == "pin" else unpin)(args.scene, args.dir))
        elif cmd == "clear":
            if args.scene is None and not args.all and not args.legacy:
                print("error: name a scene, or pass --all or --legacy", file=err)
                return 2
            if args.scene is not None and args.all:
                print("error: name a scene or pass --all, not both", file=err)
                return 2
            if args.all:
                # clear(legacy=True) without a scene clears only the legacy files.
                result = clear(cache_dir=args.dir)
                if args.legacy:
                    result["legacy"] = clear(legacy=True, cache_dir=args.dir)["legacy"]
                show(result)
            else:
                show(clear(args.scene, legacy=args.legacy, cache_dir=args.dir))
        elif cmd == "config":
            changes = {}
            if args.max_age_days is not None:
                changes["max_age_days"] = args.max_age_days
            if args.max_size_gb is not None:
                changes["max_size_gb"] = args.max_size_gb
            show(configure(cache_dir=args.dir, **changes))
    except LookupError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=err)
        return 1
    except (ValueError, CacheUnavailable) as exc:
        print(f"error: {exc}", file=err)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m world_engine_client.cache",
        description="The render cache: encrypted scene assets kept between sessions.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
