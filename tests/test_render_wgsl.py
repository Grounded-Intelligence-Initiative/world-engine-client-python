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

import os
import tempfile
import warnings
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


def test_a_platform_session_keeps_only_ciphertext_on_disk(stub, d):
    platform = DPlatform(stub)
    env = connect(stub)
    try:
        env.reset(seed=0)
        env.step([0.0] * env.action_dim)
        root = cache_root()
        assert plaintext_files(root, tempfile.gettempdir()) == []
        blobs = [p.read_bytes() for p in (root / "blobs").rglob("*") if p.is_file()]
        for url in ("/__asset/0.ply", "/__asset/1.glb"):
            assert platform.cipher(url) in blobs, f"{url} is stored as the platform sent it"
    finally:
        env.close()
    assert plaintext_files(cache_root(), tempfile.gettempdir()) == []


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
    monkeypatch.setattr(render_wgsl, "open_adapter", no_adapter)
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
    monkeypatch.setattr(render_wgsl, "open_adapter", no_adapter)

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
    monkeypatch.setattr(render_wgsl, "open_adapter", no_adapter)
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

    monkeypatch.setattr(render_wgsl, "available", lambda: True)
    monkeypatch.setattr(render_wgsl, "WGSLRenderer", D)
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
    for made in (tmp_path / "new", root.parent, root):
        assert stat.S_IMODE(made.stat().st_mode) == 0o700, made
    assert stat.S_IMODE((root / "adapters.json").stat().st_mode) == 0o600
