"""world_engine_client.assets: renderer D's read of one asset, decrypted into
memory from the render cache and checked; nothing plain on disk."""

from __future__ import annotations

import tempfile
import urllib.request

import pytest
from platform_stub import ASSETS, Platform
from test_render_cache import cache_root, plaintext_files

import world_engine_client as wec
from world_engine_client import assets, cache

pytest.importorskip("cryptography")


def _scene(stub, platform, root):
    def fetch(path, headers=None):
        req = urllib.request.Request(stub.url + "/render" + path, headers=headers or {})
        return urllib.request.urlopen(req, timeout=30)

    def open_scene():
        return cache.SceneAssets(
            fetch, platform.digests(), encrypted=platform.encrypted, platform=True, cache_dir=root
        )

    if platform.encrypted:
        s = open_scene()
    else:  # a platform from before the encryption: memory only, one warning
        with pytest.warns(RuntimeWarning, match="memory only"):
            s = open_scene()
    s.prepare()
    return s


@pytest.mark.parametrize("encrypted", [True, False])
def test_read_gives_the_asset_in_one_buffer_and_writes_nothing_plain(stub, encrypted):
    platform = Platform(stub, encrypted=encrypted)
    root = cache_root()
    s = _scene(stub, platform, root)
    try:
        for url, data in ASSETS.items():
            got = assets.read(s, url)
            assert isinstance(got, memoryview) and bytes(got) == data
        if encrypted:
            assert plaintext_files(root / "blobs", tempfile.gettempdir()) == []
    finally:
        s.close()


def test_a_cache_file_changed_after_prepare_is_refused(stub):
    platform = Platform(stub)
    root = cache_root()
    s = _scene(stub, platform, root)
    try:
        stored = platform.cipher("/__asset/1.glb")
        blob = next(
            p for p in (root / "blobs").rglob("*") if p.is_file() and p.read_bytes() == stored
        )
        data = bytearray(stored)
        data[100] ^= 0xFF  # inside the first chunk's ciphertext
        blob.write_bytes(bytes(data))
        failures = []
        for url in ASSETS:
            try:
                assets.read(s, url)
            except wec.RendererError as exc:
                failures.append(exc.code)
        assert failures == ["asset_mismatch"], failures
    finally:
        s.close()


def test_an_asset_the_server_did_not_list(stub):
    platform = Platform(stub)
    s = _scene(stub, platform, cache_root())
    try:
        with pytest.raises(wec.RendererError, match="did not list") as caught:
            assets.read(s, "/__asset/99.glb")
        assert caught.value.code == "asset_mismatch"
    finally:
        s.close()
