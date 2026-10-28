"""Several processes, one render cache (BUG-19).

Before 0.6, two renderers sharing a cache_dir wrote the same ``<digest>.part``:
one of them failed with FileNotFoundError (404 to the page, "scene loading
timed out" after 300 s), or both seemed to succeed while the file on disk mixed
both downloads, and a process killed during its download left a damaged file
the next session trusted. Each process here is a real client
(render_worker.py) whose Chrome is fake_chrome.py; the platform stub runs in
this process and sends a large splat slowly.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from platform_stub import ASSETS, ROBOT, Platform, body

import world_engine_client as wec

pytest.importorskip("websockets")
pytest.importorskip("PIL")
pytest.importorskip("cryptography")
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX signals")

HERE = Path(__file__).resolve().parent
SPLAT = "/__asset/0.ply"
BIG = {**ASSETS, SPLAT: body(b"ply\nformat binary_little_endian 1.0\n", "big", 600_000)}
EXPECTED = {url: hashlib.sha256(data).hexdigest() for url, data in {**BIG, **ROBOT}.items()}


def worker(stub, cache: Path, out: Path, mode: str, tmp: Path, *extra) -> subprocess.Popen:
    src = Path(wec.__file__).resolve().parents[1]
    path = os.pathsep.join(
        [str(src), str(HERE), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    )
    env = {**os.environ, "PYTHONPATH": path, "TMPDIR": str(tmp), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, str(HERE / "render_worker.py"), stub.url, str(cache), str(out), mode]
    return subprocess.Popen(command + list(extra), env=env)


def result(out: Path) -> dict:
    return json.loads(out.read_text())


def wait_for(condition, timeout=60.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met")
        time.sleep(0.02)


@pytest.fixture
def tmp(tmp_path):
    d = tmp_path / "proc-tmp"
    d.mkdir()
    return d


@pytest.mark.parametrize("mode", ["platform", "local"])
def test_two_processes_download_a_file_once_and_both_draw_it(stub, tmp_path, tmp, mode):
    platform = Platform(stub, encrypted=mode == "platform", assets=BIG)
    platform.delay = 0.02  # 147 pieces of 4 KiB: about 3 s for the splat
    cache = tmp_path / "shared"
    outs = [tmp_path / "a.json", tmp_path / "b.json"]
    procs = [worker(stub, cache, out, mode, tmp) for out in outs]
    for p in procs:
        assert p.wait(timeout=120) == 0
    a, b = (result(o) for o in outs)
    for r in (a, b):
        assert "error" not in r, r
        assert r["failed"] == {}, r["failed"]
        for url, digest in EXPECTED.items():
            assert r["received"][url] == digest, f"{url} came out damaged"
    assert platform.downloads[SPLAT] == 1, f"one download for both: {platform.ranges[SPLAT]}"
    first_end = min(a["ended"], b["ended"])
    assert max(a["started"], b["started"]) < first_end, "the sessions overlapped"
    leftovers = list((cache / "tmp").glob("*")) if (cache / "tmp").exists() else []
    assert leftovers == []


MIB = 1 << 20
HUGE = {**ASSETS, SPLAT: body(b"ply\nformat binary_little_endian 1.0\n", "huge", 4 * MIB)}


@pytest.mark.parametrize("mode", ["platform", "local"])
def test_a_process_killed_while_another_downloads_leaves_a_good_cache(stub, tmp_path, tmp, mode):
    """The issue list's case Z, step by step. A has written 2 MiB of the splat
    when B starts on it; B gets 1.2 MiB and is killed; then A finishes. With
    one shared ``.part`` file, B emptied it and wrote its first MiB at the
    start, so A's file had a hole of zeros, and A's page and every later
    session drew it. Now B waits for A instead."""
    import threading

    platform = Platform(stub, encrypted=mode == "platform", assets=HUGE)
    data = platform.wire(SPLAT)
    gets, b_has_written, a_may_finish = [], threading.Event(), threading.Event()

    def splat(h):
        n = len(gets)
        gets.append(h.headers.get("Range"))
        h.send_response(200)
        h.send_header("Content-Length", str(len(data)))
        h.send_header("X-We-Digest", hashlib.sha256(HUGE[SPLAT]).hexdigest())
        h.end_headers()
        try:
            if n == 0:  # A: 2.2 MiB, then wait for B to be gone
                h.wfile.write(data[: int(2.2 * MIB)])
                h.wfile.flush()
                a_may_finish.wait(60)
                h.wfile.write(data[int(2.2 * MIB) :])
            elif n == 1:  # B: 1.2 MiB, then nothing until it is killed
                h.wfile.write(data[: int(1.2 * MIB)])
                h.wfile.flush()
                b_has_written.set()
                a_may_finish.wait(60)
            else:
                h.wfile.write(data)
        except OSError:
            pass

    stub.gets["/render" + SPLAT] = splat
    cache = tmp_path / "shared"
    a = worker(stub, cache, tmp_path / "a.json", mode, tmp)
    wait_for(lambda: len(gets) >= 1)
    time.sleep(1.0)  # A reads its first two MiB
    b = worker(stub, cache, tmp_path / "b.json", mode, tmp)
    # On a cache without locks B asks for the splat as well and writes into
    # A's file; with locks it waits for A. Either way: kill B.
    b_has_written.wait(8)
    time.sleep(1.0)
    b.send_signal(signal.SIGKILL)
    b.wait(timeout=30)
    a_may_finish.set()
    assert a.wait(timeout=120) == 0
    first = result(tmp_path / "a.json")
    assert "error" not in first, first
    assert first["received"][SPLAT] == hashlib.sha256(HUGE[SPLAT]).hexdigest(), "A drew a hole"
    c = worker(stub, cache, tmp_path / "c.json", mode, tmp)
    assert c.wait(timeout=120) == 0
    later = result(tmp_path / "c.json")
    assert "error" not in later, later
    assert later["received"][SPLAT] == first["received"][SPLAT], "the next session drew a hole"
    assert len(gets) == 1, "B waited for A, and C found A's file"


@pytest.mark.parametrize("mode", ["platform", "local"])
def test_a_download_killed_halfway_is_taken_over_at_once(stub, tmp_path, tmp, mode):
    """The kernel releases a dead process's lock: the next process does not
    wait and does not trust the half-written file."""
    platform = Platform(stub, encrypted=mode == "platform", assets=BIG)
    platform.delay = 0.02
    cache = tmp_path / "shared"
    a = worker(stub, cache, tmp_path / "a.json", mode, tmp)
    wait_for(lambda: platform.sent.get(SPLAT, 0) > 300_000)
    a.send_signal(signal.SIGKILL)
    a.wait(timeout=30)
    platform.delay = 0.0
    # The day's cleanup, which also removes a dead download's file, ran
    # already: the next download of the file has to.
    import sqlite3

    con = sqlite3.connect(cache / "index.sqlite3")
    with con:
        con.execute(
            "INSERT OR REPLACE INTO meta (k, v) VALUES ('last_cleanup', ?)", (repr(time.time()),)
        )
    con.close()
    started = time.monotonic()
    c = worker(stub, cache, tmp_path / "c.json", mode, tmp)
    assert c.wait(timeout=120) == 0
    assert time.monotonic() - started < 30, "it did not wait for the dead download"
    later = result(tmp_path / "c.json")
    assert "error" not in later, later
    assert later["received"][SPLAT] == EXPECTED[SPLAT]
    assert platform.downloads[SPLAT] == 2
    parts = list((cache / "tmp").glob("*.part")) if (cache / "tmp").exists() else []
    assert parts == [], "the dead download's file was removed"


def _open_cache(root: str, start: float, results) -> None:
    from world_engine_client import cache

    time.sleep(max(0.0, start - time.time()))
    try:
        cache.RenderCache(root)
        results.put("ok")
    except Exception as exc:  # noqa: BLE001 - the test reads it
        results.put(f"{type(exc).__name__}: {exc}")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork start method")
def test_processes_that_create_one_cache_at_once_all_get_it(tmp_path):
    """Eight processes open a new cache at the same moment. Without the
    index lock most rounds had one answered "database is locked" (SQLite
    gives up at once when two connections switch the journal mode), and
    that process fell back to memory and downloaded everything again."""
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    failures = []
    for round_ in range(5):
        results = ctx.Queue()
        start = time.time() + 0.3
        root = str(tmp_path / f"cache{round_}")
        procs = [ctx.Process(target=_open_cache, args=(root, start, results)) for _ in range(8)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        failures += [r for r in (results.get(timeout=10) for _ in procs) if r != "ok"]
    assert failures == []


HOLDER = """
import os, sys, time
sys.path.insert(0, sys.argv[1])
from world_engine_client import cache
c = cache.RenderCache(sys.argv[2])
lock = c.lock(sys.argv[3])
assert lock.acquire(wait=False)
part, fd = c.new_part(sys.argv[3])
os.write(fd, b"x" * 4096)
print("held", flush=True)
time.sleep(3600)
"""


def test_a_waiter_goes_ahead_when_the_download_it_waits_for_stops(stub, tmp_path, monkeypatch):
    """A process stopped while it holds a file's lock (SIGSTOP, a debugger, a
    frozen NFS client): the kernel keeps the lock. The waiter goes ahead when
    the other's download has not grown for _STALL_SECONDS."""
    import threading

    import fake_chrome

    from world_engine_client import cache as cache_module

    platform = Platform(stub, encrypted=False)
    sim = fake_chrome.install(monkeypatch)
    monkeypatch.setattr(cache_module, "_STALL_SECONDS", 1.5)
    root = tmp_path / "shared"
    name = hashlib.sha256(ASSETS[SPLAT]).hexdigest()  # a local scene's file name: its digest
    src = Path(wec.__file__).resolve().parents[1]
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(src), str(root), name],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        holder.send_signal(signal.SIGSTOP)
        done: dict = {}

        def run():
            try:
                env = wec.connect(
                    "secret", url=stub.url, images=True, render_options={"cache_dir": str(root)}
                )
                try:
                    env.reset(seed=0)
                    done["received"] = dict(sim.received)
                finally:
                    env.close()
            except Exception as exc:  # noqa: BLE001 - the test reads it
                done["error"] = repr(exc)

        started = time.monotonic()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=60)
        waited = time.monotonic() - started
        stuck = thread.is_alive()
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=30)
    if stuck:
        thread.join(timeout=60)  # the lock is free now
    assert not stuck, "the waiter never gave up on the stopped holder"
    assert "error" not in done, done
    assert done["received"][SPLAT] == ASSETS[SPLAT]
    assert waited > 1.5, "it waited for the holder first"
    assert platform.downloads[SPLAT] == 1
