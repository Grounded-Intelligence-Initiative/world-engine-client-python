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
import warnings

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
        software = [w for w in log if "in software" in str(w.message)]
        assert len(software) == 1 and software[0].category is RuntimeWarning
        obs, _ = env.reset(seed=0)
        image = obs["rgb"]["front"]
        assert image.shape == (6, 8, 3)
        expected = np.array(COLOR) * 255
        assert np.abs(image.reshape(-1, 3).mean(axis=0) - expected).max() < 8
    finally:
        env.close()
