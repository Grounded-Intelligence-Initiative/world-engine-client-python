"""Renderer D on this machine (render_wgsl.py): render protocol 2 at connect,
the scene drawn with wgpu from the encrypted cache, no browser, and the paths
around it (the browser path for older servers and as the last fallback, the
errors when neither renders).

The tests that draw need wgpu and an adapter: a GPU, Mesa's lavapipe or the
bundled SwiftShader (world-engine-swiftshader). Without one they skip;
WE_TEST_REQUIRE_WGSL=1 turns the skip into a failure, and
WE_TEST_EXPECT_ADAPTER=<step> (system, bundled-loader, lavapipe, swiftshader,
warp) also checks which step of the adapter search rendered."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile
import warnings
import weakref
from pathlib import Path

import numpy as np
import pytest
from d_scene import CAMS, GOLDEN, DPlatform
from test_render_cache import cache_root, plaintext_files

import world_engine_client as wec
from world_engine_client import render as render_module
from world_engine_client import render_wgsl

pytest.importorskip("cryptography")
REQUIRED = os.environ.get("WE_TEST_REQUIRE_WGSL") == "1"


@pytest.fixture
def d(monkeypatch):
    """A machine where renderer D can draw (else skip): the browser path is
    made to fail loudly, so nothing passes through it unnoticed."""
    if REQUIRED:
        import wgpu  # noqa: F401
    elif not render_wgsl.available():
        pytest.skip("wgpu is not installed")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a software adapter
            render_wgsl.open_adapter(cache_dir=Path(tempfile.gettempdir()) / "probe")
    except render_wgsl.NoRenderAdapter as exc:
        if REQUIRED:
            pytest.fail(f"WE_TEST_REQUIRE_WGSL=1 but no adapter renders: {exc}")
        pytest.skip(f"no adapter renders here: {exc}")

    def no_browser(*a, **k):
        raise AssertionError("the browser path started")

    monkeypatch.setattr(render_module, "LocalRenderer", no_browser)
    return monkeypatch


def connect(stub, *, platform=True, **options):
    extra = {"access_provider": lambda: "scoped"} if platform else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a software adapter warns once
        return wec.connect(
            "secret", url=stub.url, images=True, render_options=options or None, **extra
        )


def test_a_server_with_renderer_d_draws_without_a_browser(stub, d):
    platform = DPlatform(stub)
    env = connect(stub)
    try:
        body = platform.offers[0]
        assert body["render_protocols"] == [1, 2] and body["pipelines"] == ["wgsl/1"]
        assert body["asset_encryption"] == ["dac1"]
        assert isinstance(env.renderer, render_wgsl.WGSLRenderer)
        obs, info = env.reset(seed=0)
        for cam in CAMS:
            im = obs["rgb"][cam["name"]]
            assert im.shape == (cam["h"], cam["w"], 3) and im.dtype == np.uint8, cam["name"]
            assert im.flags["C_CONTIGUOUS"] and im.std() > 2
        r = info["render"]
        assert r["renderer"] == "wgsl" and r["pipeline"] == "wgsl/1"
        assert r["adapter"] and r["adapter_type"] and r["backend"] in ("Vulkan", "Metal", "D3D12")
        assert r["color_format"] in ("rgba16float", "rgba32float", "rgb10a2unorm")
        expect = os.environ.get("WE_TEST_EXPECT_ADAPTER")
        if expect:
            assert r["step"] == expect, r
        _, _, _, _, info = env.step([0.0] * env.action_dim)
        assert info["render"]["renderer"] == "wgsl"
    finally:
        env.close()
    assert env.renderer is None


@pytest.mark.parametrize("hdri", [False, True], ids=["bake", "hdri"])
def test_a_platform_session_keeps_only_ciphertext_on_disk(stub, d, hdri):
    """The splat, the WEM1 and an HDRI (renderer D reads it, the scan knows
    its magic bytes) stay as the platform sent them."""
    import d_scene

    platform = DPlatform(stub, hdri=d_scene.hdr() if hdri else None)
    env = connect(stub)
    try:
        _, info = env.reset(seed=0)
        assert info["render"]["lighting"] == ("hdri" if hdri else "scene"), info["render"]
        env.step([0.0] * env.action_dim)
        root = cache_root()
        assert plaintext_files(root, tempfile.gettempdir()) == []
        blobs = [p.read_bytes() for p in (root / "blobs").rglob("*") if p.is_file()]
        urls = ("/__asset/0.ply", "/__asset/1.wem1") + ((d_scene.HDRI,) if hdri else ())
        for url in urls:
            assert platform.cipher(url) in blobs, f"{url} is stored as the platform sent it"
    finally:
        env.close()
    assert plaintext_files(cache_root(), tempfile.gettempdir()) == []


def test_renderer_d_downloads_what_it_draws_and_not_the_glbs(stub, d):
    """The objects arrive as the engine's WEM1; their GLBs are the browser
    path's, and a session of renderer D leaves them on the server."""
    platform = DPlatform(stub)
    env = connect(stub)
    try:
        env.reset(seed=0)
        assert isinstance(env.renderer, render_wgsl.WGSLRenderer)
        got = {u for u, n in platform.downloads.items() if n}
        assert got == {"/__asset/0.ply", "/__asset/1.wem1", "/__robot/0.bin"}, got
    finally:
        env.close()


def test_the_assets_renderer_d_reads():
    si = {
        "splat": {"url": "/__asset/0.ply"},
        "meshes": [
            {"id": "a", "url": "/__asset/1.glb", "wem": "/__asset/1.wem1"},
            {"id": "b", "box": [0.1, 0.1, 0.1]},
            {"id": "c", "url": "/__asset/3.glb"},  # no WEM1: D raises before reading it
        ],
        "hdri": {"url": "/__asset/hdri.exr"},
        "robot": {"links": [{"name": "l", "url": "/__robot/0.bin"}]},
    }
    digests = {
        u: "0" * 64
        for u in (
            "/__asset/0.ply",
            "/__asset/1.glb",
            "/__asset/1.wem1",
            "/__asset/3.glb",
            "/__asset/hdri.exr",
            "/__robot/0.bin",
            "/__asset/9.png",
        )
    }
    assert set(render_wgsl.drawn_assets(si, digests)) == {
        "/__asset/0.ply",
        "/__asset/1.wem1",
        "/__asset/hdri.exr",
        "/__robot/0.bin",
    }
    assert render_wgsl.drawn_assets({"splat": None, "meshes": []}, digests) == {}


def test_the_meshes_are_lit_and_shaded_as_the_capture_page_draws_them(stub, d):
    """wgsl/1 draws M-D2's meshes: the scene's own lighting baked at its light
    probe (the golden splats light the box), three's MeshStandardMaterial."""
    DPlatform(stub, encrypted=False)
    env = connect(stub, platform=False)
    try:
        obs, info = env.reset(seed=0)
        assert info["render"]["lighting"] in ("scene", "studio")
        lit = env.renderer._scene.info
        assert lit["pipeline"] == "wgsl/1" and lit["lighting"] == info["render"]["lighting"]
    finally:
        env.close()


def test_images_follow_the_state_and_repeat_exactly(stub, d):
    platform = DPlatform(stub, encrypted=False)  # a local world-engine serve
    env = connect(stub, platform=False)
    try:
        a, _ = env.reset(seed=0)
        b, _ = env.observe()
        for name in a["rgb"]:
            assert np.array_equal(a["rgb"][name], b["rgb"][name]), "the same state, the same image"
        platform.box = [0.25, 0.4, 0.05]  # the box moves
        c, _ = env.observe()
        assert any(not np.array_equal(a["rgb"][n], c["rgb"][n]) for n in a["rgb"])
    finally:
        env.close()


def test_the_golden_scene_matches_spark_through_the_client(stub, d):
    """Spark 2.1.0's lossless images of the golden scene (the engine's
    tests/test_wgsl_splat.py holds the host to the same bounds)."""
    from PIL import Image

    platform = DPlatform(stub, encrypted=False)
    platform.box = [50.0, 50.0, 50.0]  # out of view: the splats alone
    env = connect(stub, platform=False)
    try:
        obs, _ = env.reset(seed=0)
        tot = n = 0
        for cam in CAMS:
            ref = np.asarray(
                Image.open(GOLDEN / "spark" / f"s000_{cam['name']}.png").convert("RGB")
            )
            diff = np.abs(obs["rgb"][cam["name"]].astype(int) - ref.astype(int))
            assert diff.mean() <= 1.4 and (diff.max(-1) > 2).mean() <= 0.30, cam["name"]
            tot += diff.sum()
            n += diff.size
        assert tot / n <= 1.2
    finally:
        env.close()


# --------------------------------------------- the paths around renderer D
class FakeChrome:
    instances: list = []

    def __init__(self, fetch, scene_init, digests, **options):
        self.options = options
        self.gpu, self.software = "fake", False
        FakeChrome.instances.append(self)

    @property
    def info(self):
        return {"renderer": "chrome", "adapter": self.gpu, "software": self.software}

    def render(self, state):
        return {c["name"]: np.zeros((c["h"], c["w"], 3), np.uint8) for c in state["cameras"]}

    def close(self):
        pass


@pytest.fixture
def chrome(monkeypatch):
    FakeChrome.instances = []
    monkeypatch.setattr(render_module, "LocalRenderer", FakeChrome)
    monkeypatch.setattr(render_module, "_available_plain", lambda: True)
    monkeypatch.setattr(render_module, "available", lambda: True)
    return FakeChrome


def _choice(step, device, kind, software):
    from world_engine_client.wgsl import adapters

    return adapters.AdapterChoice(
        step=step, device=device, adapter_type=kind, backend="Vulkan", software=software
    )


GPU = _choice("system", "NVIDIA GeForce RTX 4090", "DiscreteGPU", False)
SWS = _choice("swiftshader", "SwiftShader Device (Subzero)", "CPU", True)


def no_adapter(**kw):
    raise render_wgsl.NoRenderAdapter(
        "no GPU or software adapter renders camera images here",
        code="renderer_unavailable",
        fixes=[
            "apt install libvulkan1 mesa-vulkan-drivers",
            "or pass images=False for a state-only session",
        ],
    )


def test_without_an_adapter_the_browser_path_draws(stub, chrome, monkeypatch):
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_wgsl, "find_adapter", no_adapter)
    DPlatform(stub)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        env = wec.connect("secret", url=stub.url, images=True, access_provider=lambda: "scoped")
    try:
        assert isinstance(env.renderer, FakeChrome)
        texts = [str(w.message) for w in log if issubclass(w.category, RuntimeWarning)]
        assert any("headless Chrome instead" in t for t in texts), texts
        _, info = env.reset(seed=0)
        assert info["render"]["renderer"] == "chrome"
    finally:
        env.close()


def test_when_nothing_renders_the_error_names_the_fixes(stub, chrome, monkeypatch):
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_wgsl, "find_adapter", no_adapter)

    class NoChrome(FakeChrome):
        def __init__(self, *a, **k):
            raise wec.RendererError(
                "no Chrome or Chromium to render with",
                code="renderer_unavailable",
                fixes=["python -m playwright install chromium"],
            )

    monkeypatch.setattr(render_module, "LocalRenderer", NoChrome)
    DPlatform(stub)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(render_wgsl.NoRenderAdapter) as caught:
            wec.connect("secret", url=stub.url, images=True, access_provider=lambda: "scoped")
    text = str(caught.value)
    assert "Renderer D: no GPU" in text and "Headless Chrome: no Chrome" in text
    assert caught.value.code == "renderer_unavailable"
    # Both renderers' fixes, once, in one list after the reasons; the
    # state-only way out last.
    assert caught.value.fixes == [
        "apt install libvulkan1 mesa-vulkan-drivers",
        "python -m playwright install chromium",
        "or pass images=False for a state-only session",
    ]
    assert text.count("Possible fixes") == 1
    assert text.endswith("3. or pass images=False for a state-only session")
    assert stub.paths()[-1] == "/control/release", "the controller is let go"


def test_renderer_wgsl_does_not_fall_back(stub, chrome, monkeypatch):
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_wgsl, "find_adapter", no_adapter)
    platform = DPlatform(stub)
    with pytest.raises(render_wgsl.NoRenderAdapter, match="apt install libvulkan1"):
        wec.connect(
            "secret",
            url=stub.url,
            images=True,
            access_provider=lambda: "scoped",
            render_options={"renderer": "wgsl"},
        )
    assert platform.offers[0]["render_protocols"] == [2] and FakeChrome.instances == []


def test_renderer_chrome_offers_only_the_browser_path(stub, chrome):
    platform = DPlatform(stub)
    env = wec.connect(
        "secret",
        url=stub.url,
        images=True,
        access_provider=lambda: "scoped",
        render_options={"renderer": "chrome", "chrome_args": ["--x"]},
    )
    try:
        assert (
            platform.offers[0]["render_protocols"] == [1] and "pipelines" not in platform.offers[0]
        )
        assert isinstance(env.renderer, FakeChrome)
        assert env.renderer.options["chrome_args"] == ["--x"]
    finally:
        env.close()


@pytest.fixture
def no_wgpu(stub, chrome, monkeypatch):
    """A client upgraded without the render extra (pip install -U digiarena):
    Playwright from an earlier release's extra, no wgpu."""
    from world_engine_client import env as env_module

    monkeypatch.setattr(render_wgsl, "available", lambda: False)
    monkeypatch.setattr(env_module, "_warned", set(), raising=False)
    monkeypatch.delenv("WE_RENDERER", raising=False)
    return DPlatform(stub)


def _session(stub, **options):
    """Connect, reset and close: (RuntimeWarnings, info["render"])."""
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        env = wec.connect(
            "secret",
            url=stub.url,
            images=True,
            access_provider=lambda: "scoped",
            render_options=options or None,
        )
    try:
        _, info = env.reset(seed=0)
    finally:
        env.close()
    return [str(w.message) for w in log if issubclass(w.category, RuntimeWarning)], info["render"]


def test_a_client_without_wgpu_says_why_chrome_draws(stub, no_wgpu):
    """The client offers only the browser path. It says so once per process,
    with the command that brings renderer D, and every session's
    info["render"] gives the reason."""
    upgrade = "pip install -U 'world-engine-client[render]'"
    texts, render = _session(stub)
    assert no_wgpu.offers[0]["render_protocols"] == [1]
    assert render["renderer"] == "chrome"
    assert "wgpu is not installed" in render["reason"] and upgrade in render["reason"]
    assert len(texts) == 1, texts
    assert "renderer D" in texts[0] and "render extra" in texts[0] and upgrade in texts[0]
    texts, render = _session(stub)  # the next session of this process
    assert texts == [] and upgrade in render["reason"]


@pytest.mark.parametrize("asked", ["render_options", "WE_RENDERER"])
def test_a_client_that_asks_for_chrome_hears_nothing_of_wgpu(stub, no_wgpu, monkeypatch, asked):
    options = {}
    if asked == "WE_RENDERER":
        monkeypatch.setenv("WE_RENDERER", "chrome")
    else:
        options["renderer"] = "chrome"
    texts, render = _session(stub, **options)
    assert texts == [] and render["renderer"] == "chrome" and "reason" not in render


def test_an_older_server_without_the_browser_path_installed(stub, monkeypatch):
    """A server from before renderer D answers protocol 1; this machine has no
    Playwright: say so, and how to fix it."""
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_module, "_available_plain", lambda: False)
    platform = DPlatform(stub)
    platform._connect = lambda body, base=platform._connect: _protocol(base(body), 1)
    stub.responders["/control/connect"] = platform._connect
    with pytest.raises(wec.RendererError) as caught:
        wec.connect("secret", url=stub.url, images=True, access_provider=lambda: "scoped")
    text = str(caught.value)
    assert "protocol 1" in text and "upgrade the server" in text and "images=False" in text


def _protocol(reply, protocol, pipeline=None):
    reply["render"]["protocol"] = protocol
    reply["render"].pop("pipeline", None)
    if pipeline:
        reply["render"]["pipeline"] = {"version": pipeline}
    return reply


@pytest.mark.parametrize(
    "protocol, pipeline, offered_d, match",
    [
        (2, "wgsl/1", False, "did not offer"),  # 2 without the offer
        (2, "wgsl/7", True, "pipeline 'wgsl/7'"),
        (3, None, True, "protocol 3"),
    ],
)
def test_render_protocol_answers_this_client_did_not_ask_for(
    stub, chrome, monkeypatch, protocol, pipeline, offered_d, match
):
    monkeypatch.setattr(render_wgsl, "available", lambda: offered_d)
    platform = DPlatform(stub)
    stub.responders["/control/connect"] = lambda body: _protocol(
        platform._connect(body), protocol, pipeline
    )
    with pytest.raises(wec.ProtocolError, match=match):
        wec.connect("secret", url=stub.url, images=True, access_provider=lambda: "scoped")


def test_render_options_go_to_the_renderer_that_uses_them(stub, monkeypatch):
    seen = {}

    class D:
        def __init__(self, fetch, scene_init, digests, **options):
            seen.update(options)

        def render(self, state):
            return {}

        def close(self):
            pass

    searched = {}
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_wgsl, "WGSLRenderer", D)
    monkeypatch.setattr(render_wgsl, "find_adapter", lambda **kw: searched.update(kw) or GPU)
    monkeypatch.setattr(render_wgsl, "open_adapter", lambda **kw: (object(), kw["choice"]))
    monkeypatch.setenv("WE_RENDER_ADAPTER", "name:Intel")
    DPlatform(stub)
    env = wec.connect(
        "secret",
        url=stub.url,
        images=True,
        access_provider=lambda: "scoped",
        render_options={"color_format": "rgba8unorm", "chrome_args": ["--x"], "timeout": 9},
    )
    env.close()
    assert seen["adapter"] == "name:Intel" and seen["color_format"] == "rgba8unorm"
    assert seen["timeout"] == 9 and "chrome_args" not in seen
    assert seen["opened"][1] is GPU, "the adapter the search found"
    assert searched["adapter"] == "name:Intel" and searched["timeout"] == 9


# ----------------------------------- renderer D's pipelines fail on an adapter
class GPUValidationError(Exception):
    """Stands in for wgpu's, from the wgpu package as wgpu's is."""


GPUValidationError.__module__ = "wgpu._classes"
FXC = (
    "Validation Error\n\nCaused by:\n  In wgpuDeviceCreateComputePipeline\n    Internal error: FXC "
    "D3DCompile error (Unspecified error (0x80004005)): sort(327,21-43): error X3500: array "
    "reference cannot be used as an l-value; not natively addressable"
)


class PipelineAdapter:
    """An adapter whose device builds pipelines, or raises what wgpu raises
    when FXC (D3D12) rejects a shader."""

    def __init__(self, choice, fails: bool):
        self.choice, self.fails = choice, fails

    def create_compute_pipeline(self, **kw):
        if self.fails:
            raise GPUValidationError(FXC)
        return object()

    def destroy(self):
        pass


class CoreOnAnyDevice:
    """Renderer D's core as far as these tests go: it builds a pipeline on its
    device first, as the real one does, then draws grey images."""

    color_format = "rgba16float"
    info = {"lighting": "scene"}
    last_timing: dict = {}

    def __init__(self, scene_init, read, *, device, **kw):
        device.create_compute_pipeline(layout="auto", compute={"entry_point": "sort_scatter"})

    def render(self, state):
        return {c["name"]: np.full((c["h"], c["w"], 3), 128, np.uint8) for c in state["cameras"]}

    def close(self):
        pass


@pytest.fixture
def pipelines_fail(monkeypatch):
    """A Windows machine as the adapter search sees it: a GPU on D3D12, then
    WARP. Renderer D's pipelines fail on the adapters whose names the test puts
    in the returned set (the GPU's at first). Nothing here needs wgpu."""
    from world_engine_client.wgsl import adapters
    from world_engine_client.wgsl import render_wgsl as core

    gpu = adapters.AdapterChoice(
        step="system", device="NVIDIA GeForce RTX 4090", adapter_type="DiscreteGPU", backend="D3D12"
    )
    warp = adapters.AdapterChoice(
        step="warp",
        device="Microsoft Basic Render Driver",
        adapter_type="CPU",
        backend="D3D12",
        software=True,
        pick="fallback",
    )
    failing = {gpu.device}

    def search(**kw):
        for c in (gpu, warp):
            if not adapters.failed(c):
                return c
        raise render_wgsl.NoRenderAdapter(
            "no GPU or software adapter renders camera images here (system: "
            f"{gpu.device}: {adapters.PASSED_OVER}; warp: {warp.device}: {adapters.PASSED_OVER})",
            code="renderer_unavailable",
            fixes=["or pass images=False for a state-only session"],
        )

    monkeypatch.setattr(adapters, "_failed", {})
    monkeypatch.setattr(adapters, "_warned", set())
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_wgsl, "find_adapter", search)
    monkeypatch.setattr(
        render_wgsl,
        "open_adapter",
        lambda choice=None, **kw: (PipelineAdapter(choice, choice.device in failing), choice),
    )
    monkeypatch.setattr(core, "create_device", lambda adapter: adapter)
    monkeypatch.setattr(core, "WgslRenderer", CoreOnAnyDevice)
    return failing


def test_where_d_fails_on_the_gpu_it_draws_with_the_next_adapter(stub, pipelines_fail):
    """FXC rejects a shader on the GPU's D3D12 adapter: connect() draws with
    WARP, the search's next adapter, with one warning that names the GPU and
    the compiler's error; the GPU is passed over from then on."""
    from world_engine_client.wgsl import adapters

    env, texts = _connect_d(stub)
    try:
        assert isinstance(env.renderer, render_wgsl.WGSLRenderer)
        assert env.renderer.choice.device == "Microsoft Basic Render Driver"
        assert len(texts) == 1, texts
        assert "NVIDIA GeForce RTX 4090 (DiscreteGPU, D3D12)" in texts[0]
        assert "error X3500" in texts[0] and "Microsoft Basic Render Driver" in texts[0]
        assert "images=False" in texts[0], "it says the CPU draws, once"
        obs, info = env.reset(seed=0)
        assert info["render"]["renderer"] == "wgsl" and info["render"]["software"]
        assert "error X3500" in info["render"]["reason"]
        assert all(im.mean() == 128 for im in obs["rgb"].values())
    finally:
        env.close()
    failed = [c for c in adapters._failed]
    assert failed == [("NVIDIA GeForce RTX 4090", "D3D12", "DiscreteGPU")]


def test_where_d_fails_on_every_adapter_the_browser_path_draws(stub, chrome, pipelines_fail):
    pipelines_fail.add("Microsoft Basic Render Driver")
    env, texts = _connect_d(stub)
    try:
        assert isinstance(env.renderer, FakeChrome)
        assert len(texts) == 1, texts
        assert "headless Chrome instead" in texts[0]
        assert "NVIDIA GeForce RTX 4090" in texts[0] and "Microsoft Basic Render Driver" in texts[0]
        assert "error X3500" in texts[0]
    finally:
        env.close()


def test_a_search_that_takes_the_failed_adapter_again_does_not_loop(
    stub, chrome, pipelines_fail, monkeypatch
):
    gpu = render_wgsl.find_adapter()  # the fixture's search: the GPU first
    monkeypatch.setattr(render_wgsl, "find_adapter", lambda **kw: gpu)
    env, texts = _connect_d(stub)
    try:
        assert isinstance(env.renderer, FakeChrome)
        assert len(texts) == 1 and "found no other adapter" in texts[0], texts
    finally:
        env.close()


def test_where_d_fails_on_every_adapter_without_the_browser_path_connect_says_why(
    stub, chrome, pipelines_fail
):
    pipelines_fail.add("Microsoft Basic Render Driver")
    with pytest.raises(render_wgsl.NoRenderAdapter) as err:
        _connect_d(stub, renderer="wgsl")
    text = str(err.value)
    assert "NVIDIA GeForce RTX 4090" in text and "Microsoft Basic Render Driver" in text
    assert "error X3500" in text and "pip install -U" in text and FakeChrome.instances == []


def test_open_adapter_says_what_failed_and_how_to_fix_it(monkeypatch, tmp_path):
    from world_engine_client.wgsl import adapters

    def fail(**kw):
        raise adapters.NoAdapterError(
            "nothing here", ["apt install libvulkan1 mesa-vulkan-drivers"], []
        )

    monkeypatch.setattr(adapters, "open_adapter", fail)
    with pytest.raises(render_wgsl.NoRenderAdapter) as caught:
        render_wgsl.open_adapter(cache_dir=tmp_path)
    assert caught.value.code == "renderer_unavailable" and "mesa-vulkan-drivers" in str(
        caught.value
    )
    with pytest.raises(render_wgsl.NoRenderAdapter) as caught:
        render_wgsl.open_adapter(cache_dir=tmp_path, allow_software=False)
    assert caught.value.code == "software_renderer" and "allow_software" in str(caught.value)


def test_a_software_adapter_warns_once_per_process(monkeypatch, tmp_path):
    from world_engine_client.wgsl import adapters

    choice = adapters.AdapterChoice(
        step="lavapipe", device="llvmpipe", adapter_type="CPU", backend="Vulkan", software=True
    )
    seen = []
    monkeypatch.setattr(
        adapters, "open_adapter", lambda **kw: (seen.append(kw) or object(), choice)
    )
    monkeypatch.setattr(adapters, "_warned", set())
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        render_wgsl.open_adapter(cache_dir=tmp_path)
        render_wgsl.open_adapter(cache_dir=tmp_path)
    texts = [str(w.message) for w in log if issubclass(w.category, RuntimeWarning)]
    assert len(texts) == 1 and "llvmpipe" in texts[0] and "images=False" in texts[0]
    assert seen[0]["cache_file"] == tmp_path / "adapters.json", "kept next to the render cache"


def test_the_adapter_search_keeps_its_result_in_a_private_cache(monkeypatch, tmp_path):
    """The adapter cache sits in the render cache's directory: created for this
    user only, like the render cache creates it, before anything is written."""
    import stat

    from world_engine_client.wgsl import adapters

    picked = {"device": "gpu", "adapter_type": "DiscreteGPU", "backend_type": "Vulkan"}
    monkeypatch.setattr(
        adapters,
        "run_probe",
        lambda opts, timeout=None: {"ok": True, "adapters": [], "picked": {"adapter": picked}},
    )
    monkeypatch.setattr(adapters, "_open", lambda c: object())
    monkeypatch.setattr(adapters, "instance_created", lambda: False)
    root = tmp_path / "new" / "digiarena" / "render"
    _, choice = render_wgsl.open_adapter(cache_dir=root)
    assert choice.step == "system" and (root / "adapters.json").is_file()
    if os.name == "nt":
        return  # no POSIX modes; what is under the user's profile is private by its ACL
    for made in (tmp_path / "new", root.parent, root):
        assert stat.S_IMODE(made.stat().st_mode) == 0o700, made
    assert stat.S_IMODE((root / "adapters.json").stat().st_mode) == 0o600


# ------------------------------------- scenes D cannot draw, servers that say so
def compressed_ply(n=256) -> bytes:
    """A SuperSplat compressed PLY (Spark draws these; wgsl/1 does not read them)."""
    props = ("min_x", "min_y", "min_z", "max_x", "max_y", "max_z")
    head = (
        "ply\nformat binary_little_endian 1.0\nelement chunk 1\n"
        + "".join(f"property float {p}\n" for p in props)
        + f"element vertex {n}\n"
        + "".join(f"property uint packed_{p}\n" for p in ("position", "rotation", "scale", "color"))
        + "end_header\n"
    )
    return head.encode() + bytes(4 * len(props) + 16 * n)


def test_a_scene_d_cannot_read_is_drawn_by_the_browser_path(stub, chrome, monkeypatch):
    """A server that answered protocol 2 for a splat D does not read (an
    engine that looked only at the URL's suffix): Chrome draws it, with a
    warning that says why, instead of connect failing on asset_mismatch."""
    import d_scene

    from world_engine_client.wgsl import adapters

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            render_wgsl.open_adapter(cache_dir=Path(tempfile.gettempdir()) / "probe")
    except render_wgsl.NoRenderAdapter as exc:
        if REQUIRED:
            pytest.fail(f"WE_TEST_REQUIRE_WGSL=1 but no adapter renders: {exc}")
        pytest.skip(f"no adapter renders here: {exc}")
    # renderer D starts here whatever the adapter (on a CPU adapter the client
    # would first try Chrome on the GPU: tested below)
    monkeypatch.setattr(render_wgsl, "chrome_may_be_faster", lambda *a, **k: False)
    # On a CPU adapter of a machine that shows a GPU (lavapipe here, by
    # VK_ICD_FILENAMES), the client would try Chrome on that GPU before D
    # sees the scene: that case has tests of its own below.
    monkeypatch.setattr(adapters, "gpu_devices", lambda: [])
    monkeypatch.setitem(d_scene.ASSETS, "/__asset/0.ply", compressed_ply())
    platform = DPlatform(stub, encrypted=False)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        env = wec.connect("secret", url=stub.url, images=True)
    try:
        assert platform.offers[0]["render_protocols"] == [1, 2]
        assert isinstance(env.renderer, FakeChrome)
        texts = [str(w.message) for w in log if issubclass(w.category, RuntimeWarning)]
        assert any("headless Chrome instead" in t and "chunk" in t for t in texts), texts
        _, info = env.reset(seed=0)
        assert info["render"]["renderer"] == "chrome" and "chunk" in info["render"]["reason"]
    finally:
        env.close()


def _d_opens():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            render_wgsl.open_adapter(cache_dir=Path(tempfile.gettempdir()) / "probe")
    except render_wgsl.NoRenderAdapter as exc:
        if REQUIRED:
            pytest.fail(f"WE_TEST_REQUIRE_WGSL=1 but no adapter renders: {exc}")
        pytest.skip(f"no adapter renders here: {exc}")


def _wem1_record(change) -> bytes:
    """The test box's WEM1 with its JSON record changed by ``change(doc)``."""
    import json as _json
    import struct

    import d_scene

    data = d_scene.box_wem1()
    _, jn, bn = struct.unpack("<III", data[4:16])
    doc = _json.loads(data[16 : 16 + jn])
    change(doc)
    js = _json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    return b"WEM1" + struct.pack("<III", 1, len(js), bn) + js + data[16 + jn :]


def _wem1_without(key: str) -> bytes:
    """The test box's WEM1 with a field of its JSON record left out."""
    return _wem1_record(lambda doc: doc.pop(key))


WEM1S = {
    # a WEM1 record without its primitives
    "broken wem": lambda: _wem1_without("primitives"),
    # a GLB with GPU instances: three draws each instance, D would draw one
    "instanced": lambda: _wem1_record(
        lambda doc: doc.update(unsupported=["EXT_mesh_gpu_instancing"])
    ),
    # a WEM1 cut off inside its header
    "truncated wem": lambda: b"WEM1\x01\x00\x00\x00",
}


@pytest.mark.parametrize(
    "change, why",
    [
        ("no wem", "no WEM1"),  # an engine that sent protocol 2 without the object's WEM1
        ("broken wem", "primitives"),
        ("instanced", "EXT_mesh_gpu_instancing"),
        ("truncated wem", "unpack requires"),
    ],
)
def test_a_mesh_d_cannot_draw_is_drawn_by_the_browser_path(stub, chrome, monkeypatch, change, why):
    """Renderer D leaves no object out: an object it cannot draw, or would draw
    otherwise than three (mesh_data.UNDRAWN), makes the scene the browser
    path's, with a warning that says why."""
    import d_scene

    _d_opens()
    # renderer D starts here whatever the adapter (on a CPU adapter the client
    # would first try Chrome on the GPU: tested below)
    monkeypatch.setattr(render_wgsl, "chrome_may_be_faster", lambda *a, **k: False)
    if change in WEM1S:
        monkeypatch.setitem(d_scene.ASSETS, "/__asset/1.wem1", WEM1S[change]())
    platform = DPlatform(stub, encrypted=False)
    if change == "no wem":
        base = platform.scene_init

        def without_wem():
            si = base()
            si["meshes"] = [{k: v for k, v in m.items() if k != "wem"} for m in si["meshes"]]
            return si

        platform.scene_init = without_wem
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        env = wec.connect("secret", url=stub.url, images=True)
    try:
        assert platform.offers[0]["render_protocols"] == [1, 2]
        assert isinstance(env.renderer, FakeChrome)
        texts = [str(w.message) for w in log if issubclass(w.category, RuntimeWarning)]
        assert any("headless Chrome instead" in t and why in t for t in texts), texts
    finally:
        env.close()


def test_a_server_that_declines_d_for_a_scene_says_why(stub, chrome, monkeypatch):
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    reason = "pipeline wgsl/1 does not read the scene's splat PLY: element 'chunk' first"
    platform = DPlatform(stub)
    stub.responders["/control/connect"] = lambda body: _declined(platform._connect(body), reason)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        env = wec.connect("secret", url=stub.url, images=True, access_provider=lambda: "scoped")
    try:
        assert isinstance(env.renderer, FakeChrome)
        texts = [str(w.message) for w in log if issubclass(w.category, RuntimeWarning)]
        assert any(reason in t and "not with renderer D" in t for t in texts), texts
        _, info = env.reset(seed=0)
        assert reason in info["render"]["reason"]
    finally:
        env.close()


def _declined(reply, reason):
    reply["render"].update(protocol=1, protocol_reason=reason)
    reply["render"].pop("pipeline", None)
    return reply


def test_a_client_without_the_browser_path_hears_why_the_server_declined(stub, monkeypatch):
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_module, "_available_plain", lambda: False)
    reason = "the scene has an HDRI, which pipeline wgsl/1 does not draw"
    platform = DPlatform(stub)
    stub.responders["/control/connect"] = lambda body: _declined(platform._connect(body), reason)
    with pytest.raises(wec.RendererError) as caught:
        wec.connect("secret", url=stub.url, images=True, access_provider=lambda: "scoped")
    text = str(caught.value)
    assert reason in text and "upgrade the server" not in text
    assert "playwright" in text and "images=False" in text


# ---------------------------------- a GPU that Vulkan misses and Chrome reaches
class ChromeOnTheGPU(FakeChrome):
    def __init__(self, fetch, scene_init, digests, **options):
        super().__init__(fetch, scene_init, digests, **options)
        self.gpu = "ANGLE (Intel, Mesa Intel(R) UHD Graphics (TGL GT1), OpenGL 4.6)"


class ChromeInSoftware(FakeChrome):
    def __init__(self, fetch, scene_init, digests, **options):
        FakeChrome.instances.append(self)
        self.options = options
        assert options.get("allow_software") is False
        raise wec.RendererError("software renderer (SwiftShader)", code="software_renderer")


@pytest.fixture
def vulkan_misses_the_gpu(chrome, monkeypatch, tmp_path):
    """D's search ends on the bundled SwiftShader; the machine shows a DRM
    render node. Records which D steps ran."""
    from world_engine_client.wgsl import adapters

    ran = []
    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_wgsl, "find_adapter", lambda **kw: ran.append("search") or SWS)

    def opened(**kw):
        ran.append("open")
        return object(), kw["choice"]

    class D:
        def __init__(self, *a, **kw):
            ran.append("draw")

        info = {"renderer": "wgsl"}

        def render(self, state):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(render_wgsl, "open_adapter", opened)
    monkeypatch.setattr(render_wgsl, "WGSLRenderer", D)
    monkeypatch.setattr(adapters, "gpu_devices", lambda: ["/dev/dri/renderD128"])
    monkeypatch.setattr(adapters, "nvidia_present", lambda: False)
    monkeypatch.setenv("DIGIARENA_CACHE_DIR", str(tmp_path / "cache"))
    return ran


def _connect_d(stub, **options):
    DPlatform(stub)
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        env = wec.connect(
            "secret",
            url=stub.url,
            images=True,
            access_provider=lambda: "scoped",
            render_options=options or None,
        )
    return env, [str(w.message) for w in log if issubclass(w.category, RuntimeWarning)]


def test_where_vulkan_misses_the_gpu_chrome_on_the_gpu_draws(
    stub, vulkan_misses_the_gpu, monkeypatch
):
    """A machine whose GPU has no Vulkan driver (no mesa-vulkan-drivers, a
    virtual GPU): D would draw on the CPU at about a step per second, while
    Chrome reaches the GPU through OpenGL. Measured with Vulkan hidden on the
    RTX 4090 machine: D on SwiftShader 0.9 steps/s, Chrome on the GPU's GL 15."""
    monkeypatch.setattr(render_module, "LocalRenderer", ChromeOnTheGPU)
    env, texts = _connect_d(stub)
    try:
        assert isinstance(env.renderer, ChromeOnTheGPU)
        assert env.renderer.options["allow_software"] is False, "only Chrome on the GPU"
        assert vulkan_misses_the_gpu == ["search"], "D is not opened"
        assert any(
            "Vulkan reaches no GPU" in t and "OpenGL" in t and "mesa-vulkan-drivers" in t
            for t in texts
        ), texts
        _, info = env.reset(seed=0)
        assert "Vulkan reaches no GPU" in info["render"]["reason"]
    finally:
        env.close()


class ChromeThatBreaks(FakeChrome):
    def __init__(self, fetch, scene_init, digests, **options):
        FakeChrome.instances.append(self)
        raise RuntimeError("Target page, context or browser has been closed")


@pytest.mark.parametrize("chrome_kind", [ChromeInSoftware, ChromeThatBreaks])
def test_where_chrome_draws_in_software_too_d_draws_and_this_is_kept(
    stub, vulkan_misses_the_gpu, monkeypatch, chrome_kind
):
    """Chrome on software, or a Chrome that fails to start in any way: the
    check costs one start, renderer D draws, and the next session here skips
    the check."""
    monkeypatch.setattr(render_module, "LocalRenderer", chrome_kind)
    env, _ = _connect_d(stub)
    env.close()
    assert vulkan_misses_the_gpu == ["search", "open", "draw"] and len(FakeChrome.instances) == 1
    env, _ = _connect_d(stub)  # the next session does not start Chrome to find out
    env.close()
    assert len(FakeChrome.instances) == 1 and vulkan_misses_the_gpu[-1] == "draw"


@pytest.mark.parametrize(
    "options",
    [{"renderer": "wgsl"}, {"adapter": "software"}, {"adapter": "name:SwiftShader"}],
)
def test_asking_for_d_or_for_a_software_adapter_keeps_d(
    stub, vulkan_misses_the_gpu, monkeypatch, options
):
    monkeypatch.setattr(render_module, "LocalRenderer", ChromeOnTheGPU)
    env, _ = _connect_d(stub, **options)
    env.close()
    assert FakeChrome.instances == [] and vulkan_misses_the_gpu == ["search", "open", "draw"]


def test_a_machine_that_shows_no_gpu_does_not_try_chrome(stub, vulkan_misses_the_gpu, monkeypatch):
    from world_engine_client.wgsl import adapters

    monkeypatch.setattr(adapters, "gpu_devices", lambda: [])
    monkeypatch.setattr(render_module, "LocalRenderer", ChromeOnTheGPU)
    env, _ = _connect_d(stub)
    env.close()
    assert FakeChrome.instances == [] and vulkan_misses_the_gpu == ["search", "open", "draw"]


def test_a_gpu_adapter_does_not_try_chrome(stub, vulkan_misses_the_gpu, monkeypatch):
    monkeypatch.setattr(render_wgsl, "find_adapter", lambda **kw: GPU)
    monkeypatch.setattr(render_module, "LocalRenderer", ChromeOnTheGPU)
    env, _ = _connect_d(stub)
    env.close()
    assert FakeChrome.instances == [] and vulkan_misses_the_gpu == ["open", "draw"]


# ----------------------------------- a closed renderer lets its device go at once
def _process_size() -> tuple[float, int, int] | None:
    """This process's RSS (MiB), threads and file descriptors; None off Linux."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return (
            pages * os.sysconf("SC_PAGE_SIZE") / 2**20,
            len(os.listdir("/proc/self/task")),
            len(os.listdir("/proc/self/fd")),
        )
    except (OSError, ValueError, AttributeError):
        return None


class _Collections:
    """Counts the garbage collections that run while it is installed."""

    def __init__(self):
        self.count = 0

    def __call__(self, phase, info):
        if phase == "start":
            self.count += 1

    def __enter__(self):
        self.enabled = gc.isenabled()
        gc.collect()
        gc.disable()  # only an explicit collection runs from here on
        gc.callbacks.append(self)
        return self

    def __exit__(self, *exc):
        gc.callbacks.remove(self)
        if self.enabled:
            gc.enable()


SESSIONS, WARM_UP = 8, 3


def test_a_closed_session_lets_its_device_go_at_once(stub, d):
    """Attach and release, again and again, with the garbage collector off:
    each close() frees the session's adapter, device and queue without a
    collection, and the process does not grow. In wgpu-py 0.32 a device and
    its queue refer to each other, so up to 0.7.0 they lived on until Python's
    collector ran; this scene then grew the process by about 43 MiB a session
    on lavapipe, and by 80 MiB, 4 threads and 10 file descriptors on an RTX
    4090."""
    DPlatform(stub, encrypted=False)
    sizes = []
    with _Collections() as collections:
        for i in range(SESSIONS):
            env = connect(stub, platform=False)
            renderer = env.renderer
            assert isinstance(renderer, render_wgsl.WGSLRenderer)
            device = renderer.device
            refs = {
                "adapter": weakref.ref(device.adapter),
                "device": weakref.ref(device),
                "queue": weakref.ref(device.queue),
            }
            del device
            env.observe()
            before = collections.count
            env.close()
            assert collections.count == before, "close() needed a garbage collection"
            left = [name for name, ref in refs.items() if ref() is not None]
            assert left == [], f"session {i}: the {', '.join(left)} outlived close()"
            renderer.close()  # again: nothing to do
            with pytest.raises(wec.RendererError, match="closed"):
                renderer.render({"cameras": CAMS})
            del renderer, env
            sizes.append(_process_size())
    if sizes[0] is None:
        return  # no /proc: the weak references are the test
    (rss0, threads0, fds0), (rss, threads, fds) = sizes[WARM_UP - 1], sizes[-1]
    per_session = (rss - rss0) / (SESSIONS - WARM_UP)
    assert per_session < 4.0, f"RSS grew by {per_session:.1f} MiB a session: {sizes}"
    assert threads <= threads0 + 3 and fds <= fds0 + 3, sizes


EXIT_SCRIPT = """
import atexit, gc, sys, warnings
import world_engine_client as wec

url, out, mode = sys.argv[1:4]
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    # renderer D only: on a CPU adapter the client would try Chrome on the GPU first
    env = wec.connect("secret", url=url, images=True, render_options={"renderer": "wgsl"})
env.observe()
assert type(env.renderer).__name__ == "WGSLRenderer", env.renderer


def close(open=open, finalizing=sys.is_finalizing):  # builtins may be gone by then
    env.close()
    with open(out, "a") as f:
        f.write("finalizing\\n" if finalizing() else "closed\\n")


if mode == "atexit":
    atexit.register(close)
else:  # closed by a finalizer the last collection at interpreter exit runs
    device = env.renderer.device  # still there after close(): no collection while finalizing

    class Closer:
        def __init__(self):
            self.me = self

        def __del__(self):
            close()

    gc.disable()
    Closer()
"""


@pytest.mark.parametrize("mode", ["atexit", "finalizer"])
def test_a_renderer_closes_at_interpreter_exit(stub, d, tmp_path, mode):
    """close() from an atexit handler, and from a finalizer while the
    interpreter shuts down, ends the process cleanly. The finalizer's device
    is still referenced after close(), which then does not collect: not
    while the interpreter is finalizing."""
    DPlatform(stub, encrypted=False)
    out = tmp_path / "closed.txt"
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), *filter(None, [env.get("PYTHONPATH")])])
    done = subprocess.run(
        [sys.executable, "-c", EXIT_SCRIPT, stub.url, str(out), mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert "Traceback" not in done.stderr and "Exception ignored" not in done.stderr, done.stderr
    want = "closed" if mode == "atexit" else "finalizing"
    assert out.read_text().split() == [want], done.stderr[-3000:]


def test_the_device_d_failed_on_goes_before_the_next_adapter_draws(
    stub, pipelines_fail, monkeypatch
):
    """When renderer D's pipelines fail on the GPU and WARP draws instead, the
    GPU's adapter and device are freed then, not at the next collection: the
    search keeps the reasons, not the exceptions, whose frames hold them."""
    opened = []
    stand_in = render_wgsl.open_adapter  # the fixture's

    def tracked(**kw):
        adapter, choice = stand_in(**kw)
        opened.append(weakref.ref(adapter))
        return adapter, choice

    monkeypatch.setattr(render_wgsl, "open_adapter", tracked)
    with _Collections():
        env, texts = _connect_d(stub)
        try:
            assert env.renderer.choice.device == "Microsoft Basic Render Driver"
            assert len(opened) == 2 and "error X3500" in texts[0], texts
            assert opened[0]() is None, "the device D failed on outlived the fallback"
        finally:
            env.close()
        assert opened[1]() is None


def test_the_device_of_a_scene_d_cannot_draw_goes_before_chrome_draws(stub, chrome, monkeypatch):
    """A scene renderer D cannot draw goes to the browser path; the device D
    opened for it is freed then, not at the next collection."""
    from world_engine_client.wgsl import render_wgsl as core

    devices = []

    class Device:
        def destroy(self):
            pass

    def create_device(adapter):
        device = Device()
        devices.append(weakref.ref(device))
        return device

    class Undrawable:
        def __init__(self, scene_init, read, *, device, **kw):
            raise ValueError("a mesh renderer D does not draw")

    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_wgsl, "find_adapter", lambda **kw: GPU)
    monkeypatch.setattr(render_wgsl, "open_adapter", lambda choice=None, **kw: (Device(), choice))
    monkeypatch.setattr(core, "create_device", create_device)
    monkeypatch.setattr(core, "WgslRenderer", Undrawable)
    with _Collections():
        env, texts = _connect_d(stub)
        try:
            assert isinstance(env.renderer, FakeChrome)
            assert any("a mesh renderer D does not draw" in t for t in texts), texts
            assert len(devices) == 1 and devices[0]() is None, "D's device outlived the fallback"
        finally:
            env.close()
