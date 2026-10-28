"""Client rendering without a browser: the extra requirement, the render_state
hand-off, credentials on asset requests and the digest cache."""

import asyncio
import copy
import hashlib
import importlib
import json
import re
import sys
import time
import warnings
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from conftest import ENV, OBS

import world_engine_client as wec
from world_engine_client import render as render_module

env_module = importlib.import_module("world_engine_client.env")

CAMERAS = {"front": {"width": 8, "height": 6}}
STATE = {"type": "state", "frame_id": 0, "bodies": [], "robot": {}, "cameras": []}
BUNDLE = {
    "protocol": 1,
    "scene_init": {"type": "scene_init", "cameras": []},
    "digests": {"/__asset/0.ply": "ab" * 32},
}


class FakeRenderer:
    instances = []

    def __init__(self, fetch, scene_init, digests, **options):
        self.fetch, self.scene_init, self.digests = fetch, scene_init, digests
        self.options = options
        self.states, self.closed, self.fail = [], False, False
        FakeRenderer.instances.append(self)

    def render(self, state):
        if self.fail:
            raise render_module.RendererError("GPU lost", code="renderer_unavailable")
        self.states.append(state)
        return {"front": np.full((6, 8, 3), len(self.states), np.uint8)}

    def close(self):
        self.closed = True


@pytest.fixture
def client_render(stub, monkeypatch):
    """A stub that offers client rendering; LocalRenderer replaced by a fake."""
    FakeRenderer.instances = []
    monkeypatch.setattr(render_module, "available", lambda: True)
    monkeypatch.setattr(render_module, "LocalRenderer", FakeRenderer)
    env = {**copy.deepcopy(ENV), "cameras": CAMERAS, "render": "client"}
    stub.responders["/control/connect"] = lambda body: {
        "ok": True,
        "session_id": "sess",
        "proto": 3,
        "instance_id": "inst",
        "episode_id": "e0",
        "env": env,
        **({"render": BUNDLE} if body.get("render") == "client" else {}),
    }
    for path in ("/control/reset", "/control/observe"):
        stub.responders[path] = lambda body: {
            "ok": True,
            "obs": dict(OBS),
            "info": {"episode_id": "e1", "step_id": 0},
            "render_state": STATE,
        }
    stub.responders["/control/step"] = lambda body: {
        "ok": True,
        "obs": dict(OBS),
        "info": {"episode_id": "e1", "step_id": 1},
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "render_state": STATE,
    }
    return stub


def test_images_are_rendered_locally(client_render):
    env = wec.connect("secret", url=client_render.url, images=True, render_options={"x": 1})
    connect = client_render.calls[0][1]
    assert connect["render"] == "client" and connect["images"] is True
    assert connect["asset_encryption"] == ["dac1"], "this client reads encrypted assets"
    fake = FakeRenderer.instances[0]
    assert fake.scene_init == BUNDLE["scene_init"] and fake.options.pop("x") == 1
    assert fake.options == {
        "bundle": BUNDLE,
        "platform": False,
        "origin": client_render.url,
        "scene_id": None,
    }
    obs, _ = env.reset(seed=0)
    assert obs["rgb"]["front"].shape == (6, 8, 3) and fake.states == [STATE]
    assert env.render() is obs["rgb"]["front"]
    env.close()
    assert fake.closed and client_render.paths()[-1] == "/control/release"


@pytest.mark.parametrize("kw", [{"images": True}, {"cameras": ["front"]}])
def test_images_without_the_extra_fail_before_any_request(stub, monkeypatch, kw):
    from world_engine_client import render_wgsl

    monkeypatch.setattr(render_module, "available", lambda: False)
    monkeypatch.setattr(render_module, "_available_plain", lambda: False)
    monkeypatch.setattr(render_wgsl, "available", lambda: False)
    with pytest.raises(wec.RendererError, match=r"render\]"):
        wec.connect("secret", url=stub.url, **kw)
    assert stub.calls == []


@pytest.mark.parametrize("mode", ["server", "browser"])
def test_servers_never_render(stub, mode):
    with pytest.raises(ValueError, match="no longer render"):
        wec.connect("secret", url=stub.url, images=True, render=mode)
    assert stub.calls == []


def test_a_server_without_client_rendering_is_released(client_render):
    client_render.responders["/control/connect"] = lambda body: {
        "ok": True,
        "session_id": "sess",
        "proto": 3,
        "instance_id": "inst",
        "episode_id": "e0",
        "env": {**copy.deepcopy(ENV), "cameras": CAMERAS},
    }
    with pytest.raises(wec.ProtocolError, match="predates client rendering"):
        wec.connect("secret", url=client_render.url, images=True)
    assert client_render.paths()[-1] == "/control/release"


def test_a_render_failure_leaves_physics_and_observe_renders_again(client_render):
    env = wec.connect("secret", url=client_render.url, images=True)
    env.reset(seed=0)
    fake = FakeRenderer.instances[0]
    fake.fail = True
    with pytest.raises(wec.RendererError, match="GPU lost"):
        env.observe()
    fake.fail = False
    obs, _ = env.observe()
    assert obs["rgb"]["front"][0, 0, 0] == 2
    obs, *_ = env.step([0, 0, 0, 0])  # the failed render blocked no action
    assert obs["rgb"]["front"][0, 0, 0] == 3
    env.close()


def test_an_observation_without_poses_is_a_protocol_error(client_render):
    env = wec.connect("secret", url=client_render.url, images=True)
    client_render.responders["/control/observe"] = lambda body: {
        "ok": True,
        "obs": dict(OBS),
        "info": {"episode_id": "e1", "step_id": 0},
    }
    with pytest.raises(wec.ProtocolError, match="malformed observation"):
        env.observe()
    env.close()


class Lease:
    """The server's controller lease (90 s in remote_control.py), shortened: a
    controller that sends nothing for ``seconds`` loses its session."""

    PATHS = ("/control/heartbeat", "/control/reset", "/control/step", "/control/observe")

    def __init__(self, stub, seconds):
        self.seconds, self.last = seconds, None
        connect = stub.responders["/control/connect"]

        def on_connect(body):
            self.last = time.monotonic()
            return connect(body)

        stub.responders["/control/connect"] = on_connect
        for path in self.PATHS:
            stub.responders[path] = self._checked(stub, path, stub.responders.get(path))

    def _checked(self, stub, path, reply):
        def handler(body):
            if self.last is None or time.monotonic() - self.last > self.seconds:
                self.last = None
                return {
                    "ok": False,
                    "code": "not_connected",
                    "error": "no controller session; call connect first",
                }
            self.last = time.monotonic()
            return reply(body) if reply else stub.default(path, body)

        return handler


def test_heartbeats_keep_the_lease_while_the_renderer_starts(client_render, monkeypatch):
    """A software start of a large scene can take longer than the lease. The
    heartbeat runs from connect on, so the first reset still has a session."""
    lease = Lease(client_render, seconds=1.5)
    monkeypatch.setattr(env_module, "_HEARTBEAT_SECONDS", 0.1, raising=False)

    class SlowStart(FakeRenderer):
        def __init__(self, *args, **options):
            time.sleep(2 * lease.seconds)
            super().__init__(*args, **options)

    monkeypatch.setattr(render_module, "LocalRenderer", SlowStart)
    env = wec.connect("secret", url=client_render.url, images=True)
    try:
        during_start = client_render.paths()
        obs, _ = env.reset(seed=0)
        assert obs["rgb"]["front"].shape == (6, 8, 3)
        env.step([0, 0, 0, 0])
        assert "/control/heartbeat" in during_start
    finally:
        env.close()


def test_a_failed_renderer_start_stops_the_heartbeat(client_render, monkeypatch):
    monkeypatch.setattr(env_module, "_HEARTBEAT_SECONDS", 0.05, raising=False)

    class FailedStart(FakeRenderer):
        def __init__(self, *args, **options):
            time.sleep(0.5)
            raise render_module.RendererError("no Chrome", code="renderer_unavailable")

    monkeypatch.setattr(render_module, "LocalRenderer", FailedStart)
    with pytest.raises(wec.RendererError, match="no Chrome"):
        wec.connect("secret", url=client_render.url, images=True)
    paths = client_render.paths()
    assert "/control/heartbeat" in paths, "heartbeats during the start"
    assert paths[-1] == "/control/release"
    time.sleep(0.3)
    assert client_render.paths() == paths, "no heartbeat after connect failed"


def test_asset_requests_carry_the_session_credentials(client_render):
    env = wec.connect(
        "secret", url=client_render.url, images=True, access_provider=lambda: "scoped"
    )
    client_render.files["/render/index.html"] = b"<html>"
    with env._open_render("/index.html") as response:
        assert response.read() == b"<html>"
    path, _, headers = client_render.calls[-1]
    assert path == "/render/index.html"
    assert headers["X-We-Control-Token"] == "secret"
    assert headers["Authorization"] == "Bearer scoped"
    with pytest.raises(FileNotFoundError):
        env._open_render("/missing.js")
    env.close()


def test_downloads_are_verified_against_their_digest(tmp_path, stub):
    from world_engine_client.cache import SceneAssets

    env = wec.connect("secret", url=stub.url)
    data = b"splat bytes"
    stub.files["/render/__asset/0.ply"] = data
    good = SceneAssets(
        env._open_render,
        {"/__asset/0.ply": hashlib.sha256(data).hexdigest()},
        encrypted=False,
        platform=False,
        cache_dir=tmp_path / "c",
    )
    good.prepare()
    assert good.read("/__asset/0.ply") == data
    bad = SceneAssets(
        env._open_render,
        {"/__asset/0.ply": "0" * 64},
        encrypted=False,
        platform=False,
        cache_dir=tmp_path / "c",
    )
    with pytest.raises(wec.RendererError, match="digest"):
        bad.prepare()
    assert [p.name for p in (tmp_path / "c" / "blobs").rglob("*") if p.is_file()] == [
        hashlib.sha256(data).hexdigest()
    ]
    assert list((tmp_path / "c" / "tmp").iterdir()) == []
    env.close()


def test_chrome_flag_sets_are_tried_in_order(monkeypatch):
    monkeypatch.delenv("WE_RENDER_CHROME_ARGS", raising=False)
    sets = render_module.candidate_args()
    assert sets and all("--headless=new" in args for args in sets)
    assert "--use-angle=swiftshader" in sets[-1], "software rendering is the last resort"
    assert not any("--use-angle=swiftshader" in args for args in sets[:-1])
    hardware = render_module.candidate_args(allow_software=False)
    assert hardware == sets[:-1]
    assert render_module.candidate_args(["--a"]) == [["--a"]]
    monkeypatch.setenv("WE_RENDER_CHROME_ARGS", "--b --c")
    assert render_module.candidate_args() == [["--b", "--c"]]


def test_software_rendering_is_allowed_by_default():
    import inspect

    default = inspect.signature(render_module.LocalRenderer).parameters["allow_software"].default
    assert default is True


class FakeChromium:
    """Playwright's chromium.launch: Chrome is absent; Chromium appears once installed."""

    def __init__(self, installed=False, broken=None):
        self.installed, self.broken, self.calls = installed, broken, []

    async def launch(self, channel=None, **_):
        self.calls.append(channel)
        if channel == "chrome":
            raise RuntimeError('Chromium distribution "chrome" is not found at /opt/google/chrome')
        if self.broken:
            raise RuntimeError(self.broken)
        if not self.installed:
            raise RuntimeError(
                "Executable doesn't exist at /root/.cache/ms-playwright/chromium/chrome\n"
                "Looks like Playwright was just installed. Run: playwright install"
            )
        return "browser"


def _renderer_with(chromium, auto_install=True):
    from types import SimpleNamespace

    renderer = object.__new__(render_module.LocalRenderer)  # no browser started
    renderer._pw = SimpleNamespace(chromium=chromium)
    renderer._auto_install = auto_install
    renderer._deadline = time.monotonic() + 60
    return renderer


@pytest.fixture
def downloads(monkeypatch):
    """subprocess.run for `playwright install chromium`, recorded and successful."""
    import asyncio  # noqa: F401 - used by the tests below

    calls = []
    monkeypatch.setattr(render_module, "_installed", False)

    def run(cmd, **_):
        calls.append(cmd)
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(render_module.subprocess, "run", run)
    return calls


def test_chromium_is_downloaded_once_when_no_browser_is_found(downloads):
    import asyncio
    import sys

    chromium = FakeChromium()
    original = render_module.install_chromium

    def install():
        original()
        chromium.installed = True

    renderer = _renderer_with(chromium)
    render_module.install_chromium, saved = install, render_module.install_chromium
    try:
        assert asyncio.run(renderer._launch(["--headless=new"])) == "browser"
        assert asyncio.run(renderer._launch(["--headless=new"])) == "browser"
    finally:
        render_module.install_chromium = saved
    assert downloads == [[sys.executable, "-m", "playwright", "install", "chromium"]]
    assert chromium.calls[:3] == ["chrome", None, None]


def test_auto_install_can_be_turned_off(downloads, monkeypatch):
    import asyncio

    with pytest.raises(wec.RendererError, match="playwright install chromium"):
        asyncio.run(_renderer_with(FakeChromium(), auto_install=False)._launch([]))
    assert downloads == []
    monkeypatch.delenv("WE_RENDER_AUTO_INSTALL", raising=False)
    assert render_module._auto_install_enabled(None) is True
    monkeypatch.setenv("WE_RENDER_AUTO_INSTALL", "0")
    assert render_module._auto_install_enabled(None) is False
    assert render_module._auto_install_enabled(True) is True


def test_missing_system_libraries_point_to_with_deps(downloads):
    import asyncio

    broken = "Host system is missing dependencies to run browsers. libnss3.so"
    with pytest.raises(wec.RendererError, match="--with-deps"):
        asyncio.run(_renderer_with(FakeChromium(installed=True, broken=broken))._launch([]))
    assert downloads == [], "a present but unusable Chromium is not downloaded again"


def test_the_download_flag_is_its_own(monkeypatch):
    """install_chromium() remembers a download in a module flag that starts
    False. Nothing else in the module may take its name: the download would
    never run, and available() would break after one. The tests above set
    the flag themselves, so they would not notice."""
    calls = []
    monkeypatch.setattr(render_module, "_installed", render_module._installed)  # restored
    monkeypatch.setattr(
        render_module.subprocess,
        "run",
        lambda cmd, **_: calls.append(cmd) or type("Done", (), {"returncode": 0})(),
    )
    assert render_module._installed is False
    render_module.install_chromium()
    render_module.install_chromium()
    assert [cmd[-3:] for cmd in calls] == [["playwright", "install", "chromium"]]
    assert isinstance(render_module.available(), bool)


def test_a_failed_download_says_how_to_do_it_by_hand(monkeypatch):
    import asyncio

    monkeypatch.setattr(render_module, "_installed", False)
    monkeypatch.setattr(
        render_module.subprocess, "run", lambda cmd, **_: type("D", (), {"returncode": 1})()
    )
    with pytest.raises(wec.RendererError, match="downloading Chromium failed"):
        asyncio.run(_renderer_with(FakeChromium())._launch([]))


# ----- Chrome flag sets, software rendering and speed, without a browser ----- #
SWIFTSHADER_GPU = (
    "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)"
)
RTX = "ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 4090/PCIe/SSE2, OpenGL ES 3.2)"
IGPU = "ANGLE (Intel, Mesa Intel(R) UHD Graphics (TGL GT1), OpenGL ES 3.2)"
WARP = "ANGLE (Microsoft, Microsoft Basic Render Driver Direct3D11 vs_5_0 ps_5_0, D3D11)"
SCENE = {"type": "scene_init", "cameras": []}


class PageSim:
    """Chrome and the capture page, simulated: the real LocalRenderer start
    runs against it. ``gpu`` is the hardware adapter (None: no GPU, so the
    hardware flag sets fall back to SwiftShader, as Chrome does under
    Playwright; with ``webgl`` False they get no WebGL at all). Pages answer
    like frontend/src/main.ts in the engine."""

    def __init__(self):
        self.gpu = None
        self.webgl = True
        self.load = 0.0  # seconds a page takes to load the scene; None: it never does
        self.first_frame = 0.0  # seconds a page takes for its first frame
        self.handler = None
        self.launched = []  # flag sets, in launch order
        self.frames = []  # (flag set, frame_id, warnings issued before the frame)
        self.log = []  # the test's recorded warnings

    async def serve(self, handler, host, port, **_):
        self.handler = handler
        return SimpleNamespace(
            sockets=[SimpleNamespace(getsockname=lambda: ("127.0.0.1", 9))],
            close=lambda: None,
            wait_closed=_done,
        )

    def playwright(self):
        async def start():
            return SimpleNamespace(chromium=SimpleNamespace(launch=self.launch), stop=_done)

        return SimpleNamespace(start=start)

    async def launch(self, channel=None, args=(), **_):
        self.launched.append(list(args))
        return FakeBrowser(self, list(args))


async def _done():
    return None


class FakeBrowser:
    def __init__(self, sim, args):
        self.sim, self.args, self.sockets = sim, args, []
        software_flags = "--use-angle=swiftshader" in args
        self.gpu = sim.gpu if sim.gpu and not software_flags else SWIFTSHADER_GPU

    async def new_page(self, **options):
        self.options = options  # extra_http_headers: the session token
        return FakePage(self)

    async def close(self):
        for sock in self.sockets:
            sock.inbox.put_nowait(None)


class FakePage:
    def __init__(self, browser):
        self.browser, self.handlers = browser, {}

    def on(self, event, handler):
        self.handlers[event] = handler

    async def goto(self, url, **_):
        browser = self.browser
        if not browser.sim.webgl and "--use-angle=swiftshader" not in browser.args:
            # The page script throws before it opens its socket.
            self.handlers["pageerror"](RuntimeError("Error creating WebGL context."))
            return
        sock = FakeSocket(self.browser, allowed="allowSoftware=1" in url)
        self.browser.sockets.append(sock)
        sock.task = asyncio.get_running_loop().create_task(self.browser.sim.handler(sock))


class FakeSocket:
    """The page's end of the socket."""

    def __init__(self, browser, allowed):
        self.browser, self.inbox, self.rendered = browser, asyncio.Queue(), 0
        self.software = bool(re.search("SwiftShader|llvmpipe|software", browser.gpu, re.I))
        self.refused = self.software and not allowed
        if self.refused:
            self._reply({"type": "error", "code": "software_renderer", "gpu": browser.gpu})

    def _reply(self, msg):
        self.inbox.put_nowait(msg if isinstance(msg, bytes) else json.dumps(msg))

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.inbox.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send(self, text):
        msg = json.loads(text)
        if self.refused:
            return
        sim = self.browser.sim
        if msg["type"] == "scene_init":
            if sim.load is None:
                return  # the page never reports ready, as when an asset keeps failing
            ready = {"type": "ready", "gpu": self.browser.gpu, "software": self.software}
            asyncio.get_running_loop().call_later(
                sim.load, self._reply, {**ready, "lighting": "studio"}
            )
        elif msg["type"] == "state":
            sim.frames.append((self.browser.args, msg["frame_id"], len(sim.log)))
            delay = sim.first_frame if self.rendered == 0 else 0.0
            self.rendered += 1
            header = json.dumps({"frame_id": msg["frame_id"], "images": []}).encode()
            frame = len(header).to_bytes(4, "little") + header
            asyncio.get_running_loop().call_later(delay, self._reply, frame)


@pytest.fixture
def chrome_sim(monkeypatch):
    """LocalRenderer with Playwright and websockets replaced by PageSim."""
    sim = PageSim()
    monkeypatch.setattr(render_module, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(serve=sim.serve))
    monkeypatch.setitem(sys.modules, "playwright", ModuleType("playwright"))
    async_api = SimpleNamespace(async_playwright=sim.playwright)
    monkeypatch.setitem(sys.modules, "playwright.async_api", async_api)
    monkeypatch.delenv("WE_RENDER_CHROME_ARGS", raising=False)
    monkeypatch.setattr(render_module, "_slow_warned", False, raising=False)
    return sim


def start(sim, **options):
    """A LocalRenderer started against ``sim``; its RuntimeWarnings land in ``sim.log``."""
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", ResourceWarning)  # objects of earlier tests
        sim.log = log
        try:
            return render_module.LocalRenderer(lambda path: None, SCENE, {}, **options)
        finally:
            sim.log = [w for w in log if issubclass(w.category, RuntimeWarning)]


def software_warnings(log):
    return [w for w in log if "in software" in str(w.message)]


def test_without_a_gpu_images_are_rendered_in_software_by_default(chrome_sim):
    renderer = start(chrome_sim, timeout=10)
    try:
        assert renderer.software and renderer.gpu == SWIFTSHADER_GPU
        assert "--use-angle=swiftshader" in renderer.chrome_args
        (warned,) = software_warnings(chrome_sim.log)
        assert warned.category is RuntimeWarning and "images=False" in str(warned.message)
    finally:
        renderer.close()


def test_allow_software_false_still_refuses_software_rendering(chrome_sim):
    with pytest.raises(wec.RendererError) as caught:
        start(chrome_sim, allow_software=False, timeout=10)
    assert caught.value.code == "software_renderer"
    assert "images=False" in str(caught.value)
    assert not any("--use-angle=swiftshader" in args for args in chrome_sim.launched)


def test_the_software_first_frame_is_not_held_to_the_hardware_probe(chrome_sim, monkeypatch):
    """The first frame of a real scene takes SwiftShader far longer than the
    probe that catches a hung GPU path; it gets the renderer timeout instead."""
    monkeypatch.setattr(render_module, "_PROBE_SECONDS", 0.05)
    chrome_sim.first_frame = 0.5
    renderer = start(chrome_sim, allow_software=True, timeout=10)
    try:
        assert renderer.software and "--use-angle=swiftshader" in renderer.chrome_args
    finally:
        renderer.close()


def test_the_software_warning_comes_before_the_first_frame(chrome_sim):
    renderer = start(chrome_sim, allow_software=True, timeout=10)
    renderer.close()
    (args, frame_id, warned_before) = chrome_sim.frames[0]
    assert "--use-angle=swiftshader" in args and frame_id == 1
    assert warned_before == 1, "the software warning is issued before the first frame starts"


def test_a_hardware_first_frame_is_still_held_to_the_probe(chrome_sim, monkeypatch):
    """A GPU path that hangs on its first frame is abandoned after the probe."""
    monkeypatch.setattr(render_module, "_PROBE_SECONDS", 0.05)
    chrome_sim.gpu, chrome_sim.first_frame = RTX, 0.5
    with pytest.raises(wec.RendererError, match="did not render within 0 s"):
        start(chrome_sim, allow_software=False, timeout=10)
    assert len(chrome_sim.launched) == len(render_module.candidate_args(allow_software=False))


def test_warp_counts_as_software(chrome_sim):
    """The page does not flag Windows' WARP adapter; the client does."""
    chrome_sim.gpu = WARP
    renderer = start(chrome_sim, timeout=10)
    try:
        assert renderer.software and len(software_warnings(chrome_sim.log)) == 1
    finally:
        renderer.close()
    with pytest.raises(wec.RendererError) as caught:
        start(chrome_sim, allow_software=False, timeout=10)
    assert caught.value.code == "software_renderer"


def render_steps(renderer, n):
    """Render ``n`` steps: [(step number, message)] of each speed warning."""
    warned = []
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", ResourceWarning)
        for step in range(1, n + 1):
            before = len(log)
            renderer.render(dict(STATE))
            warned += [(step, str(w.message)) for w in log[before:] if "slow" in str(w.message)]
    return warned


def test_slow_hardware_rendering_warns_once(chrome_sim, monkeypatch):
    monkeypatch.setattr(render_module, "_SLOW_STEPS", 1e9, raising=False)  # every rate is slow
    chrome_sim.gpu = IGPU
    renderer = start(chrome_sim, timeout=10)
    try:
        assert not renderer.software and chrome_sim.log == []
        warned = render_steps(renderer, 30)
        assert len(warned) == 1, "one speed warning"
        step, text = warned[0]
        assert step == render_module._SPEED_SKIP + render_module._SPEED_SAMPLES
        assert IGPU in text and "hardware adapter" in text and "images=False" in text
        assert re.search(r"[\d.]+ steps/s over 10 steps", text)
    finally:
        renderer.close()
    again = start(chrome_sim, timeout=10)
    try:
        assert render_steps(again, 30) == [], "one speed warning per process"
    finally:
        again.close()


def test_fast_hardware_rendering_does_not_warn(chrome_sim):
    chrome_sim.gpu = RTX
    renderer = start(chrome_sim, timeout=10)
    try:
        assert render_steps(renderer, 30) == [] and chrome_sim.log == []
    finally:
        renderer.close()


def test_software_rendering_gets_no_second_warning_for_speed(chrome_sim, monkeypatch):
    monkeypatch.setattr(render_module, "_SLOW_STEPS", 1e9, raising=False)
    renderer = start(chrome_sim, allow_software=True, timeout=10)
    try:
        assert renderer.software and len(chrome_sim.log) == 1
        assert render_steps(renderer, 30) == []
    finally:
        renderer.close()


def test_a_page_that_throws_before_connecting_is_given_up_quickly(chrome_sim, monkeypatch):
    """Without WebGL the page never opens its socket: the flag set fails after
    a short grace instead of the whole timeout, and the next one is tried."""
    import time

    monkeypatch.setattr(render_module, "_PAGE_ERROR_GRACE", 0.2, raising=False)
    chrome_sim.webgl = False
    started = time.monotonic()
    renderer = start(chrome_sim, allow_software=True, timeout=5)
    try:
        assert "--use-angle=swiftshader" in renderer.chrome_args
        assert time.monotonic() - started < 5, "a failed page is not waited on for the timeout"
    finally:
        renderer.close()


def test_a_scene_that_never_loads_fails_within_the_start_limit(chrome_sim, monkeypatch):
    """The flag sets share one limit, timeout + _START_GRACE, for loading. A
    scene that never loads does not get a full timeout per flag set, and the
    error says what the first set hit."""
    monkeypatch.setattr(render_module, "_START_GRACE", 0.3, raising=False)
    chrome_sim.gpu, chrome_sim.load = RTX, None
    started = time.monotonic()
    with pytest.raises(wec.RendererError) as caught:
        start(chrome_sim, timeout=3)
    elapsed = time.monotonic() - started
    assert 3 <= elapsed < 3 + 0.3 + 1.5, f"failed after {elapsed:.1f} s"
    assert "scene loading timed out" in str(caught.value)
    assert caught.value.code == "renderer_unavailable"


def test_a_software_first_frame_gets_the_timeout_after_a_long_load(chrome_sim, monkeypatch):
    """Loading used most of the start limit; the software first frame still
    gets the whole timeout."""
    monkeypatch.setattr(render_module, "_START_GRACE", 0.3, raising=False)
    chrome_sim.load = chrome_sim.first_frame = 1.5
    renderer = start(chrome_sim, allow_software=True, timeout=2)
    try:
        assert renderer.software and "--use-angle=swiftshader" in renderer.chrome_args
    finally:
        renderer.close()


def test_downloading_chromium_does_not_count_against_the_start(chrome_sim, monkeypatch):
    monkeypatch.setattr(render_module, "_START_GRACE", 0.3, raising=False)
    chrome_sim.gpu = RTX
    installed = []
    launch = chrome_sim.launch

    async def missing_until_installed(channel=None, args=(), **kw):
        if not installed:
            raise RuntimeError("Executable doesn't exist at /x/chrome\nRun: playwright install")
        return await launch(channel, args, **kw)

    def download():
        time.sleep(2)
        installed.append(True)

    monkeypatch.setattr(chrome_sim, "launch", missing_until_installed)
    monkeypatch.setattr(render_module, "install_chromium", download)
    renderer = start(chrome_sim, timeout=1, auto_install=True)
    try:
        assert installed and not renderer.software
    finally:
        renderer.close()


def test_close_closes_the_event_loop(chrome_sim):
    renderer = start(chrome_sim, allow_software=True, timeout=10)
    renderer.close()
    assert renderer._loop.is_closed()
    renderer.close()  # a second close is harmless
