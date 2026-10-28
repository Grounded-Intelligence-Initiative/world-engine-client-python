"""Scene assets between the server and the capture page: the render cache.

A stand-in for the platform (platform_stub.py) serves a scene; a stand-in for
Chrome (fake_chrome.py) loads it through the renderer's real loopback HTTP
server and WebSocket, the way the capture page does. The tests look at what
the page received, what the platform was asked for and what is on disk.

Task book §8.3 (the client half of asset protection), BUG-19 (cache: no
check on a hit, concurrent writers, permissions, page scripts downloaded
every time, stale temporary directories) and the client half of BUG-32 (the
loopback ports took any local process).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sqlite3
import stat
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import fake_chrome
import pytest
from platform_stub import ASSETS, MAGICS, PAGE, ROBOT, Platform

import world_engine_client as wec

pytest.importorskip("websockets")
pytest.importorskip("PIL")
pytest.importorskip("cryptography")

ALL = {**ASSETS, **ROBOT}


def connect(stub, *, platform: bool = True, **render_options):
    """A session with images; ``platform`` adds platform credentials."""
    extra = {"access_provider": lambda: "scoped"} if platform else {}
    return wec.connect(
        "secret", url=stub.url, images=True, render_options=render_options or None, **extra
    )


def session(stub, sim, *, platform: bool = True, **render_options) -> dict:
    """Run one session (connect, reset, close); what the page received."""
    sim.reset()
    env = connect(stub, platform=platform, **render_options)
    try:
        env.reset(seed=0)
        return dict(sim.received)
    finally:
        env.close()


def files_under(*roots) -> list[Path]:
    return sorted(p for root in roots for p in Path(root).rglob("*") if p.is_file())


def plaintext_files(*roots) -> list[Path]:
    """Files that start like a scene asset in the clear."""
    found = []
    for p in files_under(*roots):
        with open(p, "rb") as f:
            if f.read(8).startswith(MAGICS):
                found.append(p)
    return found


def cache_root() -> Path:
    from world_engine_client import cache

    return cache.directory()


def cache_module():
    from world_engine_client import cache

    return cache


@pytest.fixture
def sim(monkeypatch):
    return fake_chrome.install(monkeypatch)


# ----- the platform's encrypted assets --------------------------------------- #
def test_an_encrypted_session_keeps_only_ciphertext_on_disk(stub, sim):
    platform = Platform(stub)
    env = connect(stub)
    try:
        env.reset(seed=0)
        connect_body = stub.calls[0][1]
        assert connect_body["asset_encryption"] == ["dac1"]
        assert sim.failed == {}
        for url, data in ALL.items():
            assert sim.received[url] == data, f"the page got {url} decrypted"
        root = cache_root()
        # While the session runs and after it: no scene asset in the clear,
        # neither in the cache nor in the temporary directory.
        assert plaintext_files(root, tempfile.gettempdir()) == []
        blobs = {p.read_bytes(): p.name for p in files_under(root / "blobs")}
        for url in ASSETS:
            stored = platform.cipher(url)
            assert stored in blobs, f"{url} is stored exactly as the platform sent it"
            assert blobs[stored] != platform.digests()[url], "not under its public digest"
            assert stored.startswith(b"DAC1")
        # The index does not name the scene assets' digests either.
        index = (root / "index.sqlite3").read_bytes()
        for url in ASSETS:
            assert platform.digests()[url].encode() not in index
    finally:
        env.close()
    assert plaintext_files(cache_root(), tempfile.gettempdir()) == []
    assert env.renderer is None


def test_a_second_session_downloads_nothing_and_asks_for_the_key_again(stub, sim):
    platform = Platform(stub)
    first = session(stub, sim)
    second = session(stub, sim)
    assert first == second and all(second[url] == data for url, data in ALL.items())
    assert all(platform.downloads[url] == 1 for url in ALL), platform.downloads
    assert platform.key_requests == 2, "the key is asked for at every connect"
    assert len(stub.get_calls("/render/index.html")) == 2
    assert len(stub.get_calls("/render/assets/main-D-j9cs4x.js")) == 1


def test_a_revoked_key_stops_rendering_even_with_a_full_cache(stub, sim):
    platform = Platform(stub)
    session(stub, sim)
    platform.revoked = True
    sim.reset()
    with pytest.raises(wec.RendererError) as caught:
        connect(stub)
    assert caught.value.code == "asset_key_revoked"
    assert "revoked" in str(caught.value)
    assert sim.received == {} and sim.pages == [], "Chrome never started"
    assert all(platform.downloads[url] == 1 for url in ALL)
    assert stub.paths()[-1] == "/control/release"


def test_without_a_key_route_a_platform_session_fails_and_writes_nothing(stub, sim):
    """Design 16 §8.7: a platform session whose /__key answers 404 (a website
    without the route, a missing secret) writes no persistent plaintext."""
    platform = Platform(stub)
    del stub.gets["/render/__key"]
    with pytest.raises(wec.RendererError) as caught:
        connect(stub)
    assert caught.value.code == "asset_key_unavailable"
    assert platform.downloads == {} and sim.pages == [], "nothing fetched, no Chrome"
    root = cache_root()
    assert files_under(root / "blobs", root / "tmp") == []
    assert plaintext_files(root, tempfile.gettempdir()) == []


def _key_reply(platform, change: str) -> dict:
    import dac1_ref as ref

    key = platform.key()
    reply = {
        "v": 1,
        "alg": "A256GCM",
        "format": "dac1",
        "key_id": ref.key_id(key).hex(),
        "scope": "5f2b7d10-6c3e-4a8b-b1d9-0e4c7a2f6b31",
        "key": base64.b64encode(key).decode(),
    }
    if change == "another key's id":
        reply["key_id"] = ref.key_id(bytes(32)).hex()
    elif change == "version 2":
        reply["v"] = 2
    elif change == "another algorithm":
        reply["alg"] = "A128GCM"
    elif change == "no scope":
        del reply["scope"]
    elif change == "an empty scope":
        reply["scope"] = ""
    elif change == "a short key":
        reply["key"] = base64.b64encode(key[:16]).decode()
    elif change == "another format":
        reply["format"] = "dac2"
    return reply


@pytest.mark.parametrize(
    "change",
    [
        "another key's id",
        "version 2",
        "another algorithm",
        "no scope",
        "an empty scope",
        "a short key",
        "another format",
    ],
)
def test_a_key_reply_that_does_not_check_out_is_refused(stub, sim, change):
    from conftest import send_json

    platform = Platform(stub)
    stub.gets["/render/__key"] = lambda h: send_json(h, 200, _key_reply(platform, change))
    with pytest.raises(wec.RendererError) as caught:
        connect(stub)
    assert caught.value.code == "asset_key_unavailable", caught.value
    assert "not readable" in str(caught.value)
    assert platform.downloads == {} and sim.pages == []


def test_the_key_reply_the_platform_sends_is_accepted(stub, sim):
    from conftest import send_json

    platform = Platform(stub)
    stub.gets["/render/__key"] = lambda h: send_json(h, 200, _key_reply(platform, "none"))
    received = session(stub, sim)
    assert all(received[url] == data for url, data in ALL.items())


def _age_everything(days: float, *, last_cleanup_days: float) -> None:
    con = sqlite3.connect(cache_root() / "index.sqlite3")
    with con:
        then = time.time() - days * 86400
        con.execute("UPDATE blob SET last_used = ?", (then,))
        con.execute("UPDATE scene SET last_used = ?", (then,))
        con.execute(
            "INSERT OR REPLACE INTO meta (k, v) VALUES ('last_cleanup', ?)",
            (repr(time.time() - last_cleanup_days * 86400),),
        )
    con.close()


def test_a_scene_back_after_a_month_is_not_deleted_by_its_own_connect(stub, sim, monkeypatch):
    """Design 16 §8.5.5: opening a scene refreshes its last_used, so the age
    rule does not hit it. The connect started the day's cleanup before it
    touched its files: a scene unused for 31 days was deleted and downloaded
    again in full, which counts against the platform's daily limits."""
    from world_engine_client import cache

    platform = Platform(stub)
    session(stub, sim)
    other = cache.RenderCache()
    other.touch_scene("other-scope", "bob/garage", named=True, encrypted=False)
    stale = other.blob_path("e" * 64)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"x")
    other.record_blob("e" * 64, "plain", 1)
    other.link("other-scope", "e" * 64)
    _age_everything(31, last_cleanup_days=2)
    # The cleanup in the connect's thread: the order is the code's, not the
    # scheduler's.
    real = cache.RenderCache.maybe_cleanup

    def in_this_thread(self, background=True):
        return real(self, background=False)

    monkeypatch.setattr(cache.RenderCache, "maybe_cleanup", in_this_thread)
    received = session(stub, sim)
    assert all(received[url] == data for url, data in ALL.items())
    assert all(platform.downloads[url] == 1 for url in ALL), platform.downloads
    assert not stale.exists(), "the cleanup ran: another scene unused for 31 days went"
    assert [s["scene"] for s in cache.list_scenes()] == ["fixture"]


def test_close_forgets_the_key_and_what_was_kept_in_memory(stub, sim):
    Platform(stub)
    env = connect(stub)
    assets = env.renderer.assets
    assert assets.key is not None
    env.close()
    assert assets.key is None
    platform = Platform(stub, encrypted=False)
    with pytest.warns(RuntimeWarning, match="memory only"):
        env = connect(stub)
    assets = env.renderer.assets
    assert assets.read("/__asset/0.ply") == ASSETS["/__asset/0.ply"]
    env.close()
    assert assets._memory == {}
    assert platform.downloads["/__asset/0.ply"] == 1


def test_a_download_longer_than_the_start_limit_still_starts(stub, sim, monkeypatch):
    """Downloads do not count against the start (design 16 §8.6 item 1): the
    limit starts when the assets are in place."""
    from world_engine_client import render

    platform = Platform(stub)
    platform.delay = 0.25  # 27 pieces of 4 KiB: about 7 s
    monkeypatch.setattr(render, "_START_GRACE", 0.0)
    started = time.monotonic()
    received = session(stub, sim, timeout=3)
    assert time.monotonic() - started > 5, "the download took longer than the start limit"
    assert all(received[url] == data for url, data in ALL.items())


@pytest.mark.parametrize(
    "url", ["/__asset/../../outside", "__asset/0.ply", "/__asset/a b.ply", "/__asset/a?b"]
)
def test_asset_urls_are_checked_before_anything_is_downloaded(stub, sim, url):
    platform = Platform(stub, encrypted=False)
    bundle = platform.bundle
    digest = platform.digests()["/__asset/1.glb"]
    platform.bundle = lambda: {**bundle(), "digests": {**bundle()["digests"], url: digest}}
    with pytest.raises(wec.RendererError) as caught:
        connect(stub, platform=False)
    assert caught.value.code == "asset_mismatch" and "asset URL" in str(caught.value)
    assert platform.downloads == {}


def test_a_local_session_refuses_encrypted_bytes(stub, sim):
    """A plain bundle whose server sends DAC1 anyway: the client says what is
    wrong instead of reporting a damaged file."""
    platform = Platform(stub, encrypted=False)
    platform.wire = lambda url: (
        platform.cipher(url) if url.startswith("/__asset/") else (platform.assets[url])
    )
    with pytest.raises(wec.RendererError) as caught:
        connect(stub, platform=False)
    assert caught.value.code == "client_upgrade_required"
    assert "encrypted" in str(caught.value)


def test_a_warm_start_does_not_open_an_index_connection_per_file(stub, sim, monkeypatch):
    """Each index statement had a connection of its own, about 12 ms a file,
    most of it in SQLite's checkpoint when the last connection closes: a
    300-file scene took 3.6 s to start from a full cache."""
    from world_engine_client import cache

    many = {f"/__asset/{i}.glb": f"glTF mesh {i}".encode() * 50 for i in range(40)}
    Platform(stub, encrypted=False, assets=many)
    session(stub, sim, platform=False)
    opened = []
    real = cache.sqlite3.connect
    monkeypatch.setattr(cache.sqlite3, "connect", lambda *a, **k: opened.append(1) or real(*a, **k))
    session(stub, sim, platform=False)
    # The index set-up, the session, the cleanup, the page's scripts: a few,
    # however many files the scene has.
    assert len(opened) <= 12, f"{len(opened)} connections for {len(many) + 1} files"


def test_a_new_key_epoch_makes_the_cache_unreadable_and_downloads_again(stub, sim):
    platform = Platform(stub)
    session(stub, sim)
    old = {p.name for p in files_under(cache_root() / "blobs")}
    platform.epoch = "2.0"
    received = session(stub, sim)
    assert all(received[url] == data for url, data in ALL.items())
    assert all(platform.downloads[url] == 2 for url in ASSETS), "scene assets again"
    assert platform.downloads["/__robot/0.bin"] == 1, "the plain robot mesh is still good"
    # Files under the old key stay until unused for a day, then go when the
    # scene's key is fetched.
    now = {p.name for p in files_under(cache_root() / "blobs")}
    assert old < now
    con = sqlite3.connect(cache_root() / "index.sqlite3")
    with con:
        con.execute("UPDATE blob SET last_used = last_used - 2 * 86400")
    con.close()
    session(stub, sim)
    left = {p.name for p in files_under(cache_root() / "blobs")}
    assert left == now - (old - {platform.digests()["/__robot/0.bin"]}), left
    assert all(platform.downloads[url] == 2 for url in ASSETS)


@pytest.mark.parametrize("damage", ["flip a byte", "truncate", "empty", "another file"])
def test_a_damaged_cache_file_is_downloaded_again(stub, sim, damage):
    platform = Platform(stub)
    session(stub, sim)
    blobs = {p.read_bytes(): p for p in files_under(cache_root() / "blobs")}
    target = blobs[platform.cipher("/__asset/1.glb")]
    data = bytearray(target.read_bytes())
    if damage == "flip a byte":
        data[len(data) // 2] ^= 1
    elif damage == "truncate":
        del data[-100:]
    elif damage == "empty":
        data = bytearray()
    else:  # a whole valid DAC1 file of the same scene, under the wrong name
        data = bytearray(platform.cipher("/__asset/2.wem"))
    target.write_bytes(bytes(data))
    received = session(stub, sim)
    assert received["/__asset/1.glb"] == ASSETS["/__asset/1.glb"]
    assert platform.downloads["/__asset/1.glb"] == 2
    assert all(platform.downloads[u] == 1 for u in ALL if u != "/__asset/1.glb")
    assert target.read_bytes() == platform.cipher("/__asset/1.glb")


def test_a_file_the_platform_damaged_is_refused_and_not_kept(stub, sim):
    platform = Platform(stub)
    bad = bytearray(platform.cipher("/__asset/0.ply"))
    bad[200] ^= 1
    platform._ciphers[("/__asset/0.ply", platform.epoch)] = bytes(bad)
    with pytest.raises(wec.RendererError) as caught:
        connect(stub)
    assert caught.value.code == "asset_mismatch"
    assert bytes(bad) not in {p.read_bytes() for p in files_under(cache_root())}
    assert files_under(cache_root() / "tmp") == []


def test_other_bytes_under_the_right_key_are_refused(stub, sim):
    """Every tag verifies, but the plaintext is not the file the digest names
    (the gateway encrypted the wrong file): the digest check catches it."""
    import dac1_ref as ref

    platform = Platform(stub)
    url = "/__asset/1.glb"
    digest = platform.digests()[url]
    other = ASSETS["/__asset/2.wem"]
    head = ref.header(len(other), ref.CHUNK_SIZE, ref.key_id(platform.key()))
    gcm = ref.AesGcm(ref.file_key(platform.key(), digest, ref.CHUNK_SIZE))
    sealed = gcm.encrypt(ref._nonce(0), other, ref._aad(hashlib.sha256(head).digest(), 0, True))
    platform._ciphers[(url, platform.epoch)] = head + sealed
    with pytest.raises(wec.RendererError, match="digest") as caught:
        connect(stub)
    assert caught.value.code == "asset_mismatch"
    assert head + sealed not in {p.read_bytes() for p in files_under(cache_root())}


def test_an_unencrypted_scene_asset_in_an_encrypted_session_is_refused(stub, sim):
    platform = Platform(stub)
    platform._ciphers[("/__asset/0.ply", platform.epoch)] = ASSETS["/__asset/0.ply"]
    with pytest.raises(wec.RendererError, match="unencrypted") as caught:
        connect(stub)
    assert caught.value.code == "asset_mismatch"
    assert plaintext_files(cache_root(), tempfile.gettempdir()) == []


def test_a_broken_download_resumes_where_it_stopped(stub, sim):
    platform = Platform(stub)
    platform.cut["/__asset/0.ply"] = 10_000
    received = session(stub, sim)
    assert received["/__asset/0.ply"] == ASSETS["/__asset/0.ply"]
    assert platform.ranges["/__asset/0.ply"] == [None, "bytes=10000-"]
    stored = {p.read_bytes() for p in files_under(cache_root() / "blobs")}
    assert platform.cipher("/__asset/0.ply") in stored


def test_a_server_that_answers_a_range_with_the_whole_file_is_read_again(stub, sim):
    platform = Platform(stub)
    url = "/__asset/0.ply"
    cut = {"left": 1}
    plain_handler = stub.gets["/render" + url]

    def handler(h):
        if h.headers.get("Range"):  # 200 and the whole file, whatever was asked
            platform.downloads[url] += 1
            data = platform.cipher(url)
            h.send_response(200)
            h.send_header("Content-Length", str(len(data)))
            h.end_headers()
            h.wfile.write(data)
            return
        if cut["left"]:
            cut["left"] = 0
            platform.cut[url] = 5000
        plain_handler(h)

    stub.gets["/render" + url] = handler
    received = session(stub, sim)
    assert received[url] == ASSETS[url]
    assert platform.ranges[url] == [None] and platform.downloads[url] == 2
    assert platform.cipher(url) in {p.read_bytes() for p in files_under(cache_root() / "blobs")}


@pytest.mark.parametrize(
    "digest",
    ["../../../outside", "AB" * 32, "ab" * 31, "zz" * 32],
    ids=["path", "upper", "short", "hex"],
)
def test_digests_are_checked_before_anything_is_downloaded(stub, sim, tmp_path, digest):
    """A digest names the cache file: one that is not 64 lowercase hex
    characters (``../../x`` wrote outside the cache before) is refused."""
    platform = Platform(stub, encrypted=False)
    bundle = platform.bundle
    platform.bundle = lambda: {
        **bundle(),
        "digests": {**bundle()["digests"], "/__asset/1.glb": digest},
    }
    cache = tmp_path / "deep" / "a" / "b" / "cache"
    with pytest.raises(wec.RendererError) as caught:
        connect(stub, platform=False, cache_dir=str(cache))
    assert caught.value.code == "asset_mismatch"
    assert platform.downloads.get("/__asset/1.glb", 0) == 0
    assert not (tmp_path / "deep" / "outside.part").exists()


# ----- one file, two processes ---------------------------------------------- #
def _scene_assets(stub, root, platform):
    """A local engine's scene, read through SceneAssets alone (no renderer)."""
    from world_engine_client import cache

    def fetch(path, headers=None):
        request = urllib.request.Request(stub.url + "/render" + path, headers=headers or {})
        return urllib.request.urlopen(request, timeout=30)

    return cache.SceneAssets(
        fetch, platform.digests(), encrypted=False, platform=False, cache_dir=root
    )


def _as_another_process(store, name):
    """The name's lock as another process takes it: the OS file lock only."""
    lock = store.lock(name)
    lock._thread = threading.Lock()
    return lock


def test_a_file_deleted_while_a_session_looks_it_up_is_downloaded_again(
    stub, tmp_path, monkeypatch
):
    """Cleanup in another process holds a file's lock while it reads last_used
    and deletes the file. A session that touched and checked it meanwhile,
    without the lock, took it as cached, and serving it broke off later."""
    monkeypatch.setattr(cache_module().RenderCache, "maybe_cleanup", lambda self, **_: False)
    root = tmp_path / "c"
    platform = Platform(stub, encrypted=False)
    _scene_assets(stub, root, platform).prepare()
    url = "/__asset/0.ply"
    name = platform.digests()[url]
    store = cache_module().RenderCache(root)
    cleanup = _as_another_process(store, name)
    assert cleanup.acquire(wait=False)  # and last_used was old: it goes
    later = _scene_assets(stub, root, platform)
    thread = threading.Thread(target=later.prepare, daemon=True)
    thread.start()
    time.sleep(1.0)
    store.forget(name, "blob")
    cleanup.release(remove=True)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert later.read(url) == ASSETS[url], "the session's file is still there"
    assert platform.downloads[url] == 2


def test_waiters_check_a_finished_download_outside_its_lock(stub, tmp_path, monkeypatch):
    """Processes that waited for one download checked the finished file one
    after another, each holding its lock: with N environments on a cold
    cache, start-up grew with N times the file's size."""
    mod = cache_module()
    monkeypatch.setattr(mod.RenderCache, "maybe_cleanup", lambda self, **_: False)
    root = tmp_path / "c"
    platform = Platform(stub, encrypted=False)
    url = "/__asset/0.ply"
    name = platform.digests()[url]
    store = mod.RenderCache(root)
    downloader = _as_another_process(store, name)
    assert downloader.acquire(wait=False)
    checks = []
    real = mod.SceneAssets._verify

    def verify(self, entry, path):
        probe = _as_another_process(self.cache, entry.name)
        free = probe.acquire(wait=False)
        if free:
            probe.release()
        checks.append((entry.url, free))
        return real(self, entry, path)

    monkeypatch.setattr(mod.SceneAssets, "_verify", verify)
    waiter = _scene_assets(stub, root, platform)
    thread = threading.Thread(target=waiter.prepare, daemon=True)
    thread.start()
    time.sleep(0.5)
    path = store.blob_path(name)  # the other process's download, finished
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ASSETS[url])
    store.record_blob(name, "plain", len(ASSETS[url]))
    downloader.release(remove=True)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert waiter.read(url) == ASSETS[url] and url not in platform.downloads
    assert [free for u, free in checks if u == url] == [True], checks


# ----- a local engine's plain assets ---------------------------------------- #
def test_a_local_scene_is_cached_in_the_user_cache_dir_and_kept_after_close(stub, sim):
    platform = Platform(stub, encrypted=False)
    received = session(stub, sim, platform=False)
    assert all(received[url] == data for url, data in ALL.items())
    temp = Path(tempfile.gettempdir())
    assert list(temp.glob("we_render_*")) == [], "no temporary cache directory"
    expected = Path(os.environ["XDG_CACHE_HOME"]) / "digiarena" / "render"
    if sys.platform.startswith("linux"):
        assert cache_root() == expected
    names = {p.name for p in files_under(cache_root())}
    assert set(platform.digests().values()) <= names, "kept after close()"
    session(stub, sim, platform=False)
    assert all(platform.downloads[url] == 1 for url in ALL)


def _cached_file(root: Path, digest: str) -> Path:
    (found,) = [p for p in root.rglob(digest) if p.is_file()]
    return found


def test_a_tampered_plain_file_is_found_and_downloaded_again(stub, sim, tmp_path):
    """BUG-19: a cache hit was trusted. Another valid GLB put in the cache
    file's place used to be drawn instead of the scene's (red cube -> green)."""
    platform = Platform(stub, encrypted=False)
    cache = tmp_path / "keep"
    session(stub, sim, platform=False, cache_dir=str(cache))
    digest = platform.digests()["/__asset/1.glb"]
    _cached_file(cache, digest).write_bytes(b"glTF\x02\x00\x00\x00 a different, valid cube")
    received = session(stub, sim, platform=False, cache_dir=str(cache))
    assert received["/__asset/1.glb"] == ASSETS["/__asset/1.glb"]
    assert platform.downloads["/__asset/1.glb"] == 2
    assert _cached_file(cache, digest).read_bytes() == ASSETS["/__asset/1.glb"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_cache_directories_and_files_are_private(stub, sim, tmp_path):
    """BUG-19: a cache_dir was created 0755 with files 0644."""
    Platform(stub, encrypted=False)
    cache = tmp_path / "fresh" / "cache"
    old = os.umask(0o022)
    try:
        session(stub, sim, platform=False, cache_dir=str(cache))
    finally:
        os.umask(old)
    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    for p in cache.rglob("*"):
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == (0o700 if p.is_dir() else 0o600), (p, oct(mode))


def test_page_scripts_are_cached_by_their_hashed_names(stub, sim, tmp_path):
    """BUG-19: index.html and assets/*.js were downloaded at every connect
    (5.5 MB). Vite names scripts by their content, so those are kept."""
    Platform(stub, encrypted=False)
    cache = tmp_path / "keep"
    for _ in range(2):
        received = session(stub, sim, platform=False, cache_dir=str(cache))
        assert received["/assets/main-D-j9cs4x.js"] == PAGE["/render/assets/main-D-j9cs4x.js"]
    assert len(stub.get_calls("/render/index.html")) == 2, "no content hash: fetched fresh"
    assert len(stub.get_calls("/render/assets/main-D-j9cs4x.js")) == 1
    assert len(stub.get_calls("/render/assets/sceneEnvironment-uFkXfr3L.js")) == 1


def test_page_scripts_are_kept_per_server(stub, sim, tmp_path):
    from conftest import start_stub

    Platform(stub, encrypted=False)
    first = session(stub, sim, platform=False)
    other = start_stub()
    try:
        Platform(other, encrypted=False)
        other.files["/render/assets/main-D-j9cs4x.js"] = b"// another server's main\n"
        second = session(other, sim, platform=False)
    finally:
        other.close()
    assert first["/assets/main-D-j9cs4x.js"] == PAGE["/render/assets/main-D-j9cs4x.js"]
    assert second["/assets/main-D-j9cs4x.js"] == b"// another server's main\n"


def test_old_temporary_cache_directories_are_removed(stub, sim):
    """BUG-19: before 0.6 a killed process left plain assets in /tmp/we_render_*."""
    temp = Path(tempfile.gettempdir())
    old, recent = temp / "we_render_old12345", temp / "we_render_new12345"
    for d in (old, recent):
        d.mkdir()
        (d / ("ab" * 32)).write_bytes(ASSETS["/__asset/0.ply"])
    stale = time.time() - 2 * 86400
    for p in (old / ("ab" * 32), old):
        os.utime(p, (stale, stale))
    Platform(stub, encrypted=False)
    session(stub, sim, platform=False)
    deadline = time.monotonic() + 10
    while old.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not old.exists(), "unused for two days: removed"
    assert recent.exists(), "in use by another client, perhaps: kept"


def test_a_platform_session_with_plain_assets_writes_nothing_to_disk(stub, sim):
    """A platform that sends assets unencrypted (one from before the
    encryption): kept in memory only, with one warning."""
    platform = Platform(stub, encrypted=False)
    with pytest.warns(RuntimeWarning, match="memory only"):
        env = connect(stub)
    try:
        env.reset(seed=0)
        assert all(sim.received[url] == data for url, data in ALL.items())
        root = cache_root()
        on_disk = files_under(root / "blobs") if root.exists() else []
        assert on_disk == [] and plaintext_files(root, tempfile.gettempdir()) == []
        assert platform.downloads["/__asset/0.ply"] == 1
    finally:
        env.close()


# ----- the loopback ports (BUG-32) ------------------------------------------ #
def _proxy_get(env, path: str, token: str | None):
    headers = {} if token is None else {"X-World-Engine-Token": token}
    port = env.renderer._http.port
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_the_loopback_http_server_serves_only_the_page(stub, sim):
    platform = Platform(stub)
    env = connect(stub)
    try:
        env.reset(seed=0)
        (headers,) = sim.headers[-1:]
        token = headers.get("X-World-Engine-Token")
        assert token and len(token) >= 32, "the page's context carries the session token"
        before = len(stub.calls)
        for path in ("/__asset/0.ply", "/index.html", "/__key", "/assets/main-D-j9cs4x.js"):
            for sent in (None, "", token[:-1] + ("A" if token[-1] != "A" else "B")):
                status, _, data = _proxy_get(env, path, sent)
                assert status == 403 and ASSETS["/__asset/0.ply"][:16] not in data, (path, sent)
        assert len(stub.calls) == before, "nothing was fetched upstream for them"
        status, reply_headers, data = _proxy_get(env, "/__asset/0.ply", token)
        assert status == 200 and data == ASSETS["/__asset/0.ply"]
        assert reply_headers["Cache-Control"] == "no-store"
        for path in ("/__asset/not-in-the-bundle.ply", "/__key"):
            status, _, _ = _proxy_get(env, path, token)
            assert status == 404, path
        assert platform.key_requests == 1, "the page cannot ask for the key"
        assert platform.downloads["/__asset/0.ply"] == 1
    finally:
        env.close()


INTRUDER = (255, 0, 255)


def _intrude(url: str, result: dict) -> None:
    """Connect like the page and answer every state with magenta frames."""
    import websockets

    async def main():
        try:
            async with websockets.connect(url, max_size=None, open_timeout=5) as ws:
                result["connected"] = True
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "state":
                        await ws.send(fake_chrome.frame(msg, INTRUDER))
        except Exception as exc:  # noqa: BLE001 - the test reads it
            result["error"] = repr(exc)

    asyncio.run(main())


def test_another_local_process_cannot_answer_for_the_page(stub, sim):
    """BUG-32: the loopback WebSocket took any connection as the page's, and
    the next step's images came from whoever connected last."""
    Platform(stub)
    env = connect(stub)
    try:
        env.reset(seed=0)
        port = env.renderer._ws_port
        token = sim.ws_urls[-1].rsplit("/", 1)[1]
        assert len(token) >= 32, "the page's WebSocket URL ends in the token"
        for path in ("/", "/" + token[:-1], "/" + token + "x", "/ws"):
            result: dict = {}
            thread = threading.Thread(
                target=_intrude, args=(f"ws://127.0.0.1:{port}{path}", result), daemon=True
            )
            thread.start()
            time.sleep(0.5)
            obs, *_ = env.step([0, 0, 0, 0])
            pixel = tuple(int(c) for c in obs["rgb"]["front"][0, 0])
            assert pixel == sim.color, f"{path}: the images came from {pixel}"
            thread.join(timeout=5)
            assert "connected" not in result, f"{path} was accepted"
            assert "403" in result.get("error", ""), result
    finally:
        env.close()


def test_the_token_stays_out_of_messages(stub, sim):
    Platform(stub)
    env = connect(stub)
    try:
        renderer = env.renderer
        token = renderer._token
        page = sim.pages[-1]
        page.handlers["console"](
            SimpleNamespace(type="error", text=f"WebSocket to '{sim.ws_urls[-1]}' failed")
        )
        page.handlers["pageerror"](RuntimeError(f"bad {token}"))
        assert all(token not in line for line in renderer._console)
        assert any("<token>" in line for line in renderer._console)
    finally:
        env.close()
