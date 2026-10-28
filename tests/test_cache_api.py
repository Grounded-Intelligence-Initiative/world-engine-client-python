"""The render cache's retention, its API and its command line, without a
renderer: files are added the way a session adds them."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

from world_engine_client import cache

DAY = 86400


def add_scene(c: cache.RenderCache, scope: str, name: str, *, files=(("a" * 64, 1000),)):
    """A scene with plain files of the given (digest, size), as a session
    stores them."""
    c.touch_scene(scope, name, named=True, encrypted=False)
    for digest, size in files:
        path = c.blob_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        c.record_blob(digest, "plain", size)
        c.link(scope, digest)


def age(c: cache.RenderCache, days: float, name: str | None = None) -> None:
    """Pretend files (all, or the one called ``name``) and the scenes were
    last used ``days`` ago."""
    con = sqlite3.connect(c.index)
    with con:
        then = time.time() - days * DAY
        if name is None:
            con.execute("UPDATE blob SET last_used = ?", (then,))
        else:
            con.execute("UPDATE blob SET last_used = ? WHERE name = ?", (then, name))
        con.execute("UPDATE scene SET last_used = MIN(last_used, ?)", (then,))
    con.close()


def names(c: cache.RenderCache) -> set[str]:
    return {p.name for p in c.blobs.glob("*/*")}


@pytest.fixture
def store():
    return cache.RenderCache()


def test_the_default_place_is_the_user_cache_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("DIGIARENA_CACHE_DIR", raising=False)
    monkeypatch.setattr(cache.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.directory() == tmp_path / "xdg" / "digiarena" / "render"
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/ignored")
    assert cache.directory() == Path.home() / ".cache" / "digiarena" / "render"
    monkeypatch.setenv("DIGIARENA_CACHE_DIR", str(tmp_path / "mine"))
    assert cache.directory() == tmp_path / "mine"
    assert cache.directory(tmp_path / "given") == tmp_path / "given"
    monkeypatch.setattr(cache.sys, "platform", "darwin")
    monkeypatch.delenv("DIGIARENA_CACHE_DIR")
    assert cache.directory() == Path.home() / "Library" / "Caches" / "digiarena" / "render"


def test_files_unused_for_max_age_days_go_unless_their_scene_is_pinned(store):
    add_scene(store, "s1", "alice/kitchen", files=[("1" * 64, 100)])
    add_scene(store, "s2", "bob/garage", files=[("2" * 64, 100)])
    add_scene(store, "s3", "carol/lab", files=[("3" * 64, 100)])
    cache.pin("bob/garage")
    age(store, 31, "1" * 64)
    age(store, 31, "2" * 64)
    age(store, 29, "3" * 64)
    out = cache.cleanup()
    assert out["files"] == 1 and out["bytes"] == 100
    assert names(store) == {"2" * 64, "3" * 64}
    listed = {s["scene"]: s for s in cache.list_scenes()}
    assert set(listed) == {"bob/garage", "carol/lab"}, "a scene without files goes too"
    assert listed["bob/garage"]["pinned"] and not listed["carol/lab"]["pinned"]
    cache.unpin("bob/garage")
    cache.cleanup()
    assert names(store) == {"3" * 64}


def test_max_age_days_zero_keeps_everything(store):
    add_scene(store, "s1", "alice/kitchen")
    cache.configure(max_age_days=0)
    age(store, 10_000)
    cache.cleanup()
    assert names(store) == {"a" * 64}


def test_the_environment_wins_over_the_stored_setting(store, monkeypatch):
    add_scene(store, "s1", "alice/kitchen")
    cache.configure(max_age_days=0)
    age(store, 3)
    monkeypatch.setenv("DIGIARENA_CACHE_MAX_AGE_DAYS", "2")
    settings = cache.settings()
    assert settings["max_age_days"] == 2 and settings["from_environment"] == ["max_age_days"]
    cache.cleanup()
    assert names(store) == set()
    monkeypatch.setenv("DIGIARENA_CACHE_MAX_AGE_DAYS", "soon")
    with pytest.raises(ValueError, match="DIGIARENA_CACHE_MAX_AGE_DAYS"):
        cache.settings()


def test_a_file_used_in_the_last_hour_stays_whatever_the_setting(store):
    add_scene(store, "s1", "alice/kitchen")
    cache.configure(max_age_days=0.001)  # 86 s
    age(store, 0.01)  # 14 minutes
    cache.cleanup()
    assert names(store) == {"a" * 64}
    age(store, 0.05)  # 72 minutes
    cache.cleanup()
    assert names(store) == set()


def test_the_size_limit_removes_the_least_recently_used_unpinned_files(store):
    files = [(str(i) * 64, 400_000_000) for i in range(1, 4)]
    for i, f in enumerate(files):
        add_scene(store, f"s{i}", f"scene{i}", files=[(f[0], 10)])
        con = sqlite3.connect(store.index)
        with con:  # the sizes the index knows count, not the bytes on disk here
            con.execute("UPDATE blob SET size = ? WHERE name = ?", (f[1], f[0]))
        con.close()
        age(store, 3 - i, f[0])  # scene0 the oldest
    cache.pin("scene0")
    cache.configure(max_size_gb=0.9)
    cache.cleanup()
    assert names(store) == {"1" * 64, "3" * 64}, "the oldest unpinned file went"
    cache.configure(max_size_gb=None)
    assert cache.settings()["max_size_gb"] is None


def test_cleanup_runs_at_most_once_a_day(store):
    assert store.maybe_cleanup(background=False)
    assert not store.maybe_cleanup(background=False)
    assert not cache.RenderCache().maybe_cleanup(background=False), "another process too"


def test_configure_checks_its_values():
    for bad in (-1, "x", float("nan"), True):
        with pytest.raises(ValueError):
            cache.configure(max_age_days=bad)
    for bad in (0, -2, "big"):
        with pytest.raises(ValueError):
            cache.configure(max_size_gb=bad)
    out = cache.configure(max_age_days=7, max_size_gb="none")
    assert out["max_age_days"] == 7 and out["max_size_gb"] is None
    assert cache.configure()["max_age_days"] == 7, "arguments left out stay"


def test_pin_and_clear_take_a_name_or_a_scope(store):
    add_scene(store, "5f2b7d10-scope", "alice/kitchen", files=[("1" * 64, 10)])
    assert cache.pin("5f2b7d10-scope")[0]["pinned"]
    with pytest.raises(LookupError, match="no cached scene"):
        cache.pin("nobody/nothing")
    with pytest.raises(LookupError):
        cache.clear("nobody/nothing")
    out = cache.clear("alice/kitchen")
    assert out == {"scenes": 1, "files": 1, "bytes": 10}
    assert cache.list_scenes() == [] and names(store) == set()


def test_clearing_a_scene_keeps_files_another_scene_uses(store):
    shared = "c" * 64
    add_scene(store, "s1", "alice/kitchen", files=[("1" * 64, 10), (shared, 5)])
    add_scene(store, "s2", "bob/garage", files=[(shared, 5)])
    cache.clear("alice/kitchen")
    assert names(store) == {shared}
    (left,) = cache.list_scenes()
    assert left["scene"] == "bob/garage" and left["files"] == 1 and left["size"] == 5


def test_clear_without_a_scene_empties_the_cache_but_keeps_settings(store):
    add_scene(store, "s1", "alice/kitchen", files=[("1" * 64, 10)])
    add_scene(store, "s2", "bob/garage", files=[("2" * 64, 10)])
    cache.configure(max_age_days=5)
    out = cache.clear()
    assert out["scenes"] == 2 and out["files"] == 2
    assert cache.list_scenes() == [] and names(store) == set()
    assert cache.settings()["max_age_days"] == 5


def test_a_file_being_downloaded_is_not_cleared(store):
    add_scene(store, "s1", "alice/kitchen", files=[("1" * 64, 10)])
    lock = store.lock("1" * 64)
    assert lock.acquire(wait=False)
    try:
        assert cache.clear()["files"] == 0
        assert names(store) == {"1" * 64}
    finally:
        lock.release()


def test_list_scenes_reports_name_size_and_last_use(store):
    add_scene(store, "s1", "alice/kitchen", files=[("1" * 64, 10), ("2" * 64, 30)])
    (scene,) = cache.list_scenes()
    assert scene["scene"] == "alice/kitchen" and scene["scope"] == "s1"
    assert scene["files"] == 2 and scene["size"] == 40 and not scene["encrypted"]
    assert abs(scene["last_used"].timestamp() - time.time()) < 60
    assert cache.list is cache.list_scenes, "also as cache.list, like digiarena.cache.list"


def test_a_given_name_replaces_the_engines_name_but_not_the_other_way(store):
    store.touch_scene("s1", "kitchen_001", named=False, encrypted=True)
    store.touch_scene("s1", "alice/kitchen", named=True, encrypted=True)
    store.touch_scene("s1", "kitchen_001", named=False, encrypted=True)
    assert [s["scene"] for s in cache.list_scenes()] == ["alice/kitchen"]


def test_orphans_missing_files_and_old_parts_are_tidied(store):
    add_scene(store, "s1", "alice/kitchen", files=[("1" * 64, 10), ("2" * 64, 10)])
    store.blob_path("1" * 64).unlink()  # a row without its file
    orphan = store.blob_path("9" * 64)  # a file without its row, two days old
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"x")
    fresh = store.blob_path("8" * 64)  # a file without its row, just written
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_bytes(b"x")
    dead = store.tmp / f"{'7' * 32}.999999999.abcd.part"  # its process is gone
    dead.write_bytes(b"x")
    mine = store.tmp / f"{'6' * 32}.{os.getpid()}.abcd.part"  # this process writes it
    mine.write_bytes(b"x")
    stale = time.time() - 2 * DAY
    os.utime(orphan, (stale, stale))
    cache.cleanup()
    assert names(store) == {"2" * 64, "8" * 64}
    assert not dead.exists() and mine.exists()
    assert [s["files"] for s in cache.list_scenes()] == [1]


def test_old_key_files_go_once_unused_for_a_day(store):
    store.touch_scene("s1", "alice/kitchen", named=True, encrypted=True, key_id="new")
    for name, kid in (("a" * 32, "old"), ("b" * 32, "new")):
        p = store.blob_path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"DAC1")
        store.record_blob(name, "dac1", 4, key_id=kid)
        store.link("s1", name)
    assert store.drop_other_keys("s1", "new") == 0, "used today: a session may still read it"
    age(store, 2, "a" * 32)
    assert store.drop_other_keys("s1", "new") == 4
    assert names(store) == {"b" * 32}


def test_legacy_plain_files_are_removed_on_request(store, tmp_path):
    old = store.root / ("d" * 64)
    old.write_bytes(b"ply\nformat binary_little_endian 1.0\n")
    (store.root / ("e" * 64 + ".part")).write_bytes(b"glTF")
    keep = store.root / "notes.txt"
    keep.write_bytes(b"mine")
    temp = Path(tempfile.gettempdir()) / "we_render_abc"
    temp.mkdir()
    (temp / ("f" * 64)).write_bytes(b"ply\n")
    out = cache.clear(legacy=True)
    size = len(b"ply\nformat binary_little_endian 1.0\n") + len(b"glTF")
    assert out == {"legacy": {"files": 2, "bytes": size, "temp_dirs": 1}}
    assert not old.exists() and keep.exists() and not temp.exists()


def test_reading_a_cache_that_does_not_exist_creates_nothing(tmp_path):
    root = tmp_path / "never"
    assert cache.list_scenes(root) == []
    assert cache.settings(root)["max_age_days"] == cache.DEFAULT_MAX_AGE_DAYS
    assert cache.cleanup(root)["files"] == 0
    assert cache.clear(cache_dir=root)["files"] == 0
    with pytest.raises(LookupError):
        cache.pin("alice/kitchen", root)
    assert not root.exists(), "nothing was created"
    assert cache.configure(max_age_days=3, cache_dir=root)["max_age_days"] == 3
    assert (root / "index.sqlite3").exists(), "configure stores the setting"


def test_a_cache_on_a_network_file_system_warns_and_skips_wal(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_fs_type", lambda path: "nfs4")
    with pytest.warns(RuntimeWarning, match="network file system"):
        store = cache.RenderCache(tmp_path / "nfs")
    con = sqlite3.connect(store.index)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    con.close()
    assert store.network_fs


def test_without_file_locks_one_process_still_excludes_its_threads(store, monkeypatch):
    import errno as errnos

    def no_locks(fd):
        raise OSError(errnos.ENOLCK, "No locks available")

    monkeypatch.setattr(cache, "_lock_fd", no_locks)
    first = store.lock("q" * 32)
    with pytest.warns(RuntimeWarning, match="without file locks"):
        assert first.acquire(wait=False)
    assert not store.lock("q" * 32).acquire(wait=False), "the thread lock still holds"
    first.release()


def test_a_cache_dir_that_cannot_be_used_says_so(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(cache.CacheUnavailable, match="cannot be used"):
        cache.RenderCache(blocker / "cache")


def test_two_threads_never_hold_one_name(store):
    inside, overlaps = [], []

    def worker():
        for _ in range(50):
            with store.lock("n" * 32):
                inside.append(1)
                if len(inside) > 1:
                    overlaps.append(1)
                time.sleep(0.0005)
                inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == []


def test_a_lock_file_removed_while_waited_on_still_excludes(store, monkeypatch):
    """Cleanup deletes free lock files. A process that opened one just before
    and then got its lock holds a file no longer at the path: it must notice
    and lock the new file instead, or two processes would hold one name."""
    name = "m" * 32
    first = store.lock(name)
    assert first.acquire(wait=False)
    real, raced = cache._lock_fd, []

    def lock_after_the_holder_left(fd):
        if not raced:  # the waiter has opened the file; now the holder leaves
            raced.append(True)
            first.release(remove=True)
        return real(fd)

    monkeypatch.setattr(cache, "_lock_fd", lock_after_the_holder_left)
    second = store.lock(name)
    second._thread = threading.Lock()  # as if another process
    assert second.acquire()
    monkeypatch.setattr(cache, "_lock_fd", real)
    third = store.lock(name)
    third._thread = threading.Lock()
    assert not third.acquire(wait=False), "two holders of one name"
    second.release()
    assert third.acquire(wait=False)
    third.release()


# ----- command line --------------------------------------------------------- #
def run(*argv) -> tuple[int, str, str]:
    import argparse

    parser = argparse.ArgumentParser(prog="cache")
    cache.add_arguments(parser)
    out, err = io.StringIO(), io.StringIO()
    code = cache.run(parser.parse_args(list(argv)), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_the_command_line(store, tmp_path):
    add_scene(store, "s1", "alice/kitchen", files=[("1" * 64, 10)])
    code, out, _ = run("ls")
    (scene,) = json.loads(out)
    assert code == 0 and scene["scene"] == "alice/kitchen" and scene["size"] == 10
    assert json.loads(run("pin", "alice/kitchen")[1])[0]["pinned"] is True
    assert json.loads(run("unpin", "alice/kitchen")[1])[0]["pinned"] is False
    code, _, err = run("pin", "nobody")
    assert code == 1 and err.startswith("error: no cached scene")
    code, out, _ = run("config", "--max-age-days", "12", "--max-size-gb", "none")
    assert code == 0 and json.loads(out)["max_age_days"] == 12
    code, _, err = run("config", "--max-age-days", "-3")
    assert code == 1 and "0 or more" in err
    code, _, err = run("clear")
    assert code == 2 and "--all" in err
    code, out, _ = run("clear", "--all")
    assert code == 0 and json.loads(out)["files"] == 1
    other = tmp_path / "other"
    code, out, _ = run("--dir", str(other), "ls")
    assert code == 0 and json.loads(out) == [] and not other.exists()


def test_python_dash_m_runs_the_command_line(store):
    import subprocess
    import sys

    add_scene(store, "s1", "alice/kitchen")
    src = Path(cache.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(src), "DIGIARENA_CACHE_DIR": str(store.root)}
    done = subprocess.run(
        [sys.executable, "-m", "world_engine_client.cache", "ls"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)[0]["scene"] == "alice/kitchen"
