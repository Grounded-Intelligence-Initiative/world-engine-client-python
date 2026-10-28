"""Pipeline wgsl/1 as this release draws it (design doc 16 §9.3): the D test
scene (tests/d_scene.py) and three variants of it through connect() and
renderer D on Mesa's lavapipe, against the images in tests/data/wgsl_pin.

An engine answers a pipeline version for one drawing: the shaders
(tests/test_wgsl_vendored.py pins their text) and the host that feeds them
(the copy in world_engine_client.wgsl and WGSLRenderer here: lighting, bake,
materials, shadow). A change that draws otherwise is a new pipeline version
once a release ships this one; for a new version, or before that release,
record the images again:

    WE_TEST_RECORD_PIN=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \\
        python -m pytest tests/test_wgsl_pin.py

Needs wgpu and lavapipe on x86_64, else skipped (other adapters draw a little
otherwise, and lavapipe on other processors is not measured);
WE_TEST_EXPECT_ADAPTER=lavapipe (the CI's lavapipe tier) makes the skip a
failure. Per image at most one value in 5000 (at least 2) may be off by 1: on
the engine's pinned scenes Mesa 25.2.8 draws the record exactly and 25.0.7
leaves at most 2 values off by 1."""

from __future__ import annotations

import json
import os
import platform as _platform
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from d_scene import DPlatform, hdr

import world_engine_client as wec
from world_engine_client import render as render_module
from world_engine_client import render_wgsl

PIN = Path(__file__).resolve().parent / "data" / "wgsl_pin"
RECORD = os.environ.get("WE_TEST_RECORD_PIN") == "1"
REQUIRED = os.environ.get("WE_TEST_EXPECT_ADAPTER") == "lavapipe"
VARIANTS = ("scene", "studio", "hdri", "parts")
RULE = "per image, at most one value in 5000 (at least 2) off by 1, none by more"


class PinPlatform(DPlatform):
    """The D test scene, a local world-engine serve (plain assets): as it is
    ("scene": the golden splats bake the lighting, a WEM1 box, a robot link),
    without the splats ("studio"), with an HDRI ("hdri"), with design doc 14's
    local offset on the box and two proxy boxes, one a table the box's shadow
    falls on ("parts")."""

    def __init__(self, stub, variant: str):
        super().__init__(stub, encrypted=False, hdri=hdr() if variant == "hdri" else None)
        self.variant = variant

    def scene_init(self) -> dict:
        si = super().scene_init()
        if self.variant == "studio":
            si["splat"] = None
        elif self.variant == "parts":
            si["meshes"][0]["local"] = {"pos": [0.0, 0.05, 0.02], "quat": [0, 0, 0.0998, 0.995]}
            si["meshes"] += [
                {
                    "id": "plate",
                    "url": "",
                    "box": [0.05, 0.08, 0.04],
                    "pos": [-0.2, 0.0, 0.1],
                    "quat": [0.0, 0.0, 0.3826834, 0.9238795],
                },
                # a table under the box: the key light's shadow falls on it
                {"id": "table", "url": "", "box": [0.45, 0.45, 0.01], "pos": [0.25, -0.1, -0.07]},
            ]
        return si


@pytest.fixture
def lavapipe(monkeypatch):
    def skip(why):
        if REQUIRED:
            pytest.fail(f"WE_TEST_EXPECT_ADAPTER=lavapipe but {why}")
        pytest.skip(why)

    if _platform.machine().lower() not in ("x86_64", "amd64"):
        skip("the pin images are lavapipe's on x86_64")
    if not render_wgsl.available():
        skip("wgpu is not installed")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a software adapter
            render_wgsl.open_adapter(
                adapter="name:llvmpipe", cache_dir=Path(tempfile.gettempdir()) / "probe"
            )
    except render_wgsl.NoRenderAdapter as exc:
        skip(f"no lavapipe: {exc}")

    def no_browser(*a, **k):
        raise AssertionError("the browser path started")

    monkeypatch.setattr(render_module, "LocalRenderer", no_browser)


def _images(stub, variant: str) -> tuple[dict[str, np.ndarray], dict]:
    platform = PinPlatform(stub, variant)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # lavapipe is a software adapter
        env = wec.connect(
            "secret", url=stub.url, images=True, render_options={"adapter": "name:llvmpipe"}
        )
    try:
        assert isinstance(env.renderer, render_wgsl.WGSLRenderer)
        out = {}
        obs, info = env.reset(seed=0)
        for cam, img in obs["rgb"].items():
            out[f"{variant}_0_{cam}.png"] = img
        platform.box = [0.25, 0.4, 0.05]  # the box moves
        obs, _ = env.observe()
        for cam, img in obs["rgb"].items():
            out[f"{variant}_1_{cam}.png"] = img
        r = info["render"]
        assert "llvmpipe" in r["adapter"], r
        return out, {"lighting": r["lighting"], "color_format": r["color_format"]}
    finally:
        env.close()


def _off(got: np.ndarray, want: np.ndarray) -> str | None:
    if got.shape != want.shape:
        return f"shape {got.shape}, pinned {want.shape}"
    d = np.abs(got.astype(int) - want.astype(int))
    allowed = max(2, d.size // 5000)
    if d.max() > 1 or int((d > 0).sum()) > allowed:
        return (
            f"{int((d > 0).sum())} values differ (at most {allowed} may, by 1), "
            f"the largest by {int(d.max())}, mean {d.mean():.4f}/255"
        )
    return None


@pytest.mark.parametrize("variant", VARIANTS)
def test_wgsl1_draws_the_pinned_images(stub, lavapipe, variant):
    from PIL import Image

    images, lit = _images(stub, variant)
    meta_path = PIN / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if RECORD:
        PIN.mkdir(parents=True, exist_ok=True)
        for f, img in images.items():
            Image.fromarray(img).save(PIN / f)
        record = json.loads(
            (Path(render_wgsl.__file__).parent / "wgsl" / "SOURCE.json").read_text()
        )
        meta.update(pipeline=record["pipeline"], engine_commit=record["engine_commit"], rule=RULE)
        meta.setdefault("variants", {})[variant] = {"images": sorted(images), **lit}
        meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        return
    pinned = meta["variants"][variant]
    assert meta["pipeline"] == render_wgsl.PIPELINE
    assert lit == {k: pinned[k] for k in lit}, (lit, pinned)
    assert sorted(images) == pinned["images"]
    bad = {}
    for f, img in images.items():
        why = _off(img, np.asarray(Image.open(PIN / f).convert("RGB")))
        if why:
            bad[f] = why
    assert not bad, (
        f"renderer D draws pipeline {render_wgsl.PIPELINE} otherwise than this release "
        f"ships it: {bad}. Draw differently under a new pipeline version (in the engine); "
        "record again (see this file's docstring) only for a version no release has shipped."
    )
