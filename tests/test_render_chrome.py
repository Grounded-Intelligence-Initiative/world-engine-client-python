"""Software rendering in a real headless Chrome, on a small scene.

The scene is a minimal capture page served from the stub server's /render/
prefix: it speaks the capture page protocol (frontend/src/main.ts in the
engine) and draws one camera with WebGL2. Chrome runs on SwiftShader, so this
works on any machine. Skipped without the render extra or a browser, unless
WE_TEST_REQUIRE_CHROME=1 (the CI job that installs the render extra sets it).
"""

import copy
import os
import string
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import ENV, OBS

import world_engine_client as wec
from world_engine_client import render as render_module

REQUIRED = os.environ.get("WE_TEST_REQUIRE_CHROME") == "1"
if not REQUIRED:
    for _module in ("playwright", "websockets", "PIL"):
        pytest.importorskip(_module)

SWIFTSHADER = ["--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
SWIFTSHADER += ["--no-sandbox"]
FIRST_FRAME_MS = 1500  # longer than the probe patched below, far shorter than the timeout
# WE_TEST_BROWSER=shell (the CI job) runs these on this package's headless
# shell rather than an installed Google Chrome; WE_TEST_BROWSER=chrome on Chrome.
BROWSER = os.environ.get("WE_TEST_BROWSER")


@pytest.fixture(autouse=True)
def browser(monkeypatch):
    if BROWSER:
        monkeypatch.setenv("WE_RENDER_BROWSER", BROWSER)


COLOR = (0.2, 0.6, 0.4)

PAGE = string.Template("""<!doctype html>
<meta charset="utf-8">
<canvas id="c" width="8" height="6"></canvas>
<script type="module">
const params = new URLSearchParams(location.search);
const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
const info = gl.getExtension("WEBGL_debug_renderer_info");
const gpu = gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
const software = /SwiftShader|llvmpipe|software/i.test(gpu);
const refused = software && params.get("allowSoftware") !== "1";
const ws = new WebSocket(params.get("ws"));
let frames = 0;
ws.onopen = () => {
  if (refused) ws.send(JSON.stringify({ type: "error", code: "software_renderer", gpu }));
};
ws.onmessage = async (ev) => {
  const msg = JSON.parse(ev.data);
  if (refused) return;
  if (msg.type === "scene_init") {
    ws.send(JSON.stringify({ type: "ready", gpu, software, lighting: "studio" }));
    return;
  }
  if (msg.type !== "state") return;
  frames += 1;
  if (frames === 1) await new Promise((done) => setTimeout(done, $first_frame_ms));
  gl.clearColor($red, $green, $blue, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const blob = await new Promise((done) => canvas.toBlob(done, "image/jpeg", 0.95));
  const jpeg = new Uint8Array(await blob.arrayBuffer());
  const image = { name: "front", w: 8, h: 6, offset: 0, length: jpeg.length };
  const header = new TextEncoder().encode(
    JSON.stringify({ frame_id: msg.frame_id, images: [image] }));
  const frame = new Uint8Array(4 + header.length + jpeg.length);
  new DataView(frame.buffer).setUint32(0, header.length, true);
  frame.set(header, 4);
  frame.set(jpeg, 4 + header.length);
  ws.send(frame);
};
</script>
""")


@pytest.fixture
def small_scene(stub):
    """The stub server as a gym whose capture page is PAGE."""
    red, green, blue = COLOR
    page = PAGE.substitute(first_frame_ms=FIRST_FRAME_MS, red=red, green=green, blue=blue)
    stub.files["/render/index.html"] = page.encode()
    env = {**copy.deepcopy(ENV), "cameras": {"front": {"width": 8, "height": 6}}}
    bundle = {"protocol": 1, "scene_init": {"type": "scene_init", "cameras": []}, "digests": {}}
    stub.responders["/control/connect"] = lambda body: {
        "ok": True,
        "session_id": "sess",
        "proto": 3,
        "instance_id": "inst",
        "episode_id": "e0",
        "env": env,
        "render": bundle,
    }
    stub.responders["/control/reset"] = lambda body: {
        "ok": True,
        "obs": dict(OBS),
        "info": {"episode_id": "e1", "step_id": 0},
        "render_state": {"type": "state", "bodies": [], "robot": {}, "cameras": []},
    }
    return stub


def test_software_rendering_is_not_held_to_the_hardware_probe(small_scene, monkeypatch):
    monkeypatch.setattr(render_module, "_PROBE_SECONDS", 0.5)
    options = {"chrome_args": SWIFTSHADER, "allow_software": True, "auto_install": False}
    with warnings.catch_warnings(record=True) as log:
        warnings.simplefilter("always")
        try:
            env = wec.connect(
                "secret",
                url=small_scene.url,
                images=True,
                render_options={**options, "timeout": 60},
            )
        except wec.RendererError as exc:
            if "no Chrome or Chromium" in str(exc) and not REQUIRED:
                pytest.skip("no Chrome or Chromium on this machine")
            raise
    try:
        assert env.renderer.software and "SwiftShader" in env.renderer.gpu
        if BROWSER:
            assert env.renderer.info["browser"] == {"shell": "headless-shell"}.get(BROWSER, BROWSER)
        software = [w for w in log if "in software" in str(w.message)]
        assert len(software) == 1 and software[0].category is RuntimeWarning
        obs, _ = env.reset(seed=0)
        image = obs["rgb"]["front"]
        assert image.shape == (6, 8, 3)
        expected = np.array(COLOR) * 255
        assert np.abs(image.reshape(-1, 3).mean(axis=0) - expected).max() < 8
    finally:
        env.close()


# ----- encrypted scene assets, decrypted for a real Chrome ------------------- #
ASSET_PAGE = """<!doctype html>
<meta charset="utf-8">
<canvas id="c" width="8" height="6"></canvas>
<script type="module">
// Like the capture page: on scene_init fetch every asset the scene names
// (plain from the loopback server here), then draw. The first three bytes of
// the splat are the colour.
const params = new URLSearchParams(location.search);
const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
const info = gl.getExtension("WEBGL_debug_renderer_info");
const gpu = gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
const software = /SwiftShader|llvmpipe|software/i.test(gpu);
const ws = new WebSocket(params.get("ws"));
let color = [0, 0, 0];
ws.onmessage = async (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "scene_init") {
    const urls = [];
    (function walk(node) {
      if (Array.isArray(node)) node.forEach(walk);
      else if (node && typeof node === "object")
        for (const [k, v] of Object.entries(node)) k === "url" ? urls.push(v) : walk(v);
    })(msg);
    for (const url of urls) {
      const res = await fetch(url);
      if (!res.ok) throw new Error("asset " + url + ": " + res.status);
      const bytes = new Uint8Array(await res.arrayBuffer());
      if (msg.splat && url === msg.splat.url) color = [...bytes.slice(0, 3)].map((b) => b / 255);
    }
    ws.send(JSON.stringify({ type: "ready", gpu, software, lighting: "studio" }));
    return;
  }
  if (msg.type !== "state") return;
  gl.clearColor(color[0], color[1], color[2], 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const blob = await new Promise((done) => canvas.toBlob(done, "image/png"));
  const png = new Uint8Array(await blob.arrayBuffer());
  const image = { name: "front", w: 8, h: 6, offset: 0, length: png.length };
  const header = new TextEncoder().encode(
    JSON.stringify({ frame_id: msg.frame_id, images: [image] }));
  const frame = new Uint8Array(4 + header.length + png.length);
  new DataView(frame.buffer).setUint32(0, header.length, true);
  frame.set(header, 4);
  frame.set(png, 4 + header.length);
  ws.send(frame);
};
</script>
"""

SPLAT_COLOR = (30, 160, 90)


def _chrome_or_skip(exc: Exception):
    if "no Chrome or Chromium" in str(exc) and not REQUIRED:
        pytest.skip("no Chrome or Chromium on this machine")
    raise exc


def test_an_encrypted_asset_is_decrypted_for_real_chrome(stub):
    """The platform sends DAC1; the loopback server hands Chrome the
    plaintext, and only with the session token Playwright adds to the page's
    requests. The WebSocket URL carries the token too."""
    pytest.importorskip("cryptography")
    import dac1_ref
    from platform_stub import Platform

    splat = bytes(SPLAT_COLOR) + dac1_ref.pattern("colour", 70_000)
    platform = Platform(stub, assets={"/__asset/0.ply": splat})
    stub.files["/render/index.html"] = ASSET_PAGE.encode()
    options = {"chrome_args": SWIFTSHADER, "allow_software": True, "auto_install": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            env = wec.connect(
                "secret",
                url=stub.url,
                images=True,
                access_provider=lambda: "scoped",
                render_options={**options, "timeout": 60},
            )
        except wec.RendererError as exc:
            _chrome_or_skip(exc)
    try:
        obs, _ = env.reset(seed=0)
        mean = obs["rgb"]["front"].reshape(-1, 3).mean(axis=0)
        assert np.abs(mean - np.array(SPLAT_COLOR)).max() < 3, mean
        assert platform.downloads["/__asset/0.ply"] == 1 and platform.key_requests == 1
    finally:
        env.close()


def _plaintext_files(*roots):
    from platform_stub import MAGICS

    found = []
    for root in roots:
        for p in Path(root).rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    with open(p, "rb") as f:
                        if f.read(16).startswith(MAGICS):
                            found.append(p)
            except OSError:
                continue
    return found


def _kill_processes_using(path: Path) -> None:
    """Chrome processes started for ``path`` (their profile is in it)."""
    import signal

    for proc in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        if str(path).encode() in cmd and int(proc.name) != os.getpid():
            try:
                os.kill(int(proc.name), signal.SIGKILL)
            except OSError:
                pass


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="scans /proc")
def test_no_scene_asset_is_on_disk_in_the_clear_during_a_session_or_after_kill_9(stub, tmp_path):
    """A real session in its own process: while it runs and after kill -9,
    the cache and the temporary directory (where Chrome keeps its profile)
    hold no splat, GLB, WEM1, PNG or JPEG in the clear."""
    import signal
    import time

    pytest.importorskip("cryptography")
    import shutil
    import tempfile

    from platform_stub import ASSETS, Platform
    from test_cache_processes import worker

    platform = Platform(stub)
    stub.files["/render/index.html"] = ASSET_PAGE.encode()
    cache = tmp_path / "cache"
    # Chrome keeps a socket in its profile, under TMPDIR: the path must stay
    # within the Unix socket length limit, so not under pytest's tmp_path.
    tmp = Path(tempfile.mkdtemp(prefix="wec-", dir="/tmp"))
    marker, out = tmp_path / "ready", tmp_path / "out.json"
    proc = worker(stub, cache, out, "platform", tmp, "--chrome", "--hold", str(marker))
    try:
        deadline = time.monotonic() + 240
        while not marker.exists():
            if proc.poll() is not None:
                import json

                error = json.loads(out.read_text()).get("error", "")
                _chrome_or_skip(RuntimeError(error))
            assert time.monotonic() < deadline, "the session did not start"
            time.sleep(0.1)
        assert all(platform.downloads[url] == 1 for url in ASSETS), "Chrome got every asset"
        during = _plaintext_files(cache, tmp)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=30)
        time.sleep(1.0)
        after = _plaintext_files(cache, tmp)
    finally:
        if proc.poll() is None:
            proc.kill()
        _kill_processes_using(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    assert during == [], during
    assert after == [], after
    stored = {p.read_bytes() for p in (cache / "blobs").rglob("*") if p.is_file()}
    assert all(platform.cipher(url) in stored for url in ASSETS)
