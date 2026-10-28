"""The integrated-GPU bench (bench/igpu): its scenes are the same bytes on every
machine, its capture page and references are the ones it says, its numbers
and checks follow README.md, and renderer D here draws what it drew on the RTX
4090 (that one needs wgpu and an adapter; it is skipped without one)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench" / "igpu"
if not BENCH.is_dir():  # an sdist: the bench is the repository's only
    pytest.skip("bench/igpu is not here", allow_module_level=True)
sys.path.insert(0, str(BENCH))

import compare  # noqa: E402
import run  # noqa: E402
import scene  # noqa: E402
import worker  # noqa: E402

REQUIRED = os.environ.get("WE_TEST_REQUIRE_WGSL") == "1"


# ----------------------------------------------------------------- the inputs
@pytest.mark.parametrize("name", ["r2_400k", "room200k"])
def test_the_scenes_are_made_byte_for_byte(name):
    make, want = scene.SCENES[name]
    data = make()
    assert hashlib.sha256(data).hexdigest() == want
    assert scene.splat(name) == data


def test_r2_400k_is_the_round_two_splat():
    """The generator made the PLY that design doc 16's round-2 r2_400k used."""
    assert scene.SPLAT_SHA256 == "c1ab11fcbbf2adb0eed554809cdcae1ada3e80dee657db32a723fb0319839d80"
    head = scene.make_splat(n=10)[:200].decode("ascii", "replace")
    assert head.startswith("ply\nformat binary_little_endian 1.0\nelement vertex 10\n")


def test_room200k_is_a_3dgs_ply_with_sh_degree_3():
    data = scene.make_room(n=1000)
    header, _, body = data.partition(b"end_header\n")
    props = [line.split()[-1] for line in header.decode().splitlines() if line.startswith("prop")]
    assert props == scene.ROOM_PROPS and len([p for p in props if p.startswith("f_rest")]) == 45
    rows = np.frombuffer(body, "<f4").reshape(1000, len(props))
    q = rows[:, props.index("rot_0") : props.index("rot_3") + 1]
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5), "unit quaternions"
    logit = rows[:, props.index("opacity")]
    assert (logit > 0).mean() > 0.6 and (logit < 0).mean() > 0.1, "opaque and translucent"


def test_the_trace_is_splat_only():
    for name in scene.SCENES:
        scene_init, states, warmup = scene.trace(name)
        assert scene_init["scene_id"] == f"{name}_splat"
        assert scene_init["meshes"] == [] and scene_init["robot"]["links"] == []
        assert scene_init["robot"]["ee_proxy_size"] == [0.0, 0.0, 0.0]
        assert scene_init["splat"]["url"] == scene.SPLAT_URL
        assert (len(states), warmup) == (221, 21)
        names = [c["name"] for c in scene_init["cameras"]]
        assert names == ["third_person", "top_down", "wrist"]
        for st in states:
            assert [c["name"] for c in st["cameras"]] == names and st["bodies"] == []
            assert min(st["robot"]["ee_pos"]) >= 100, "the end-effector box is out of sight"
    moving = {c["name"] for st in states for c in st["cameras"] if c != states[0]["cameras"][0]}
    assert "wrist" in moving


def test_the_capture_page_is_the_one_its_manifest_names():
    manifest = run.page_manifest()  # raises when a file differs
    assert set(manifest["files"]) == {
        "index.html",
        "assets/main-C0ZK_nJr.js",
        "assets/sceneEnvironment-BecPG80m.js",
    }
    assert manifest["build_stamp"]["mode"] == "production" and len(manifest["engine_commit"]) == 40
    html = (scene.CAPTURE_PAGE / "index.html").read_text()
    for name in manifest["files"]:
        if name.startswith("assets/"):
            assert f'"/{name}"' in html


def test_the_references_are_there():
    want = {f"s{k:03d}_{cam}" for k, cam in worker.REF_IMAGES}
    for name in worker.SCENES:
        for folder in worker.REF_SETS.values():
            found = {p.stem for p in (worker.REFS / name / folder).glob("*.png")}
            assert found == want, (name, folder)
        ref = worker.refs(name)
        assert all(img.shape == (256, 256, 3) for imgs in ref.values() for img in imgs.values())
    made = json.loads((worker.REFS / "refs.json").read_text())
    for name in worker.SCENES:
        for folder in worker.REF_SETS.values():
            entry = made["sets"][name][folder]
            assert "RTX 4090" in entry["adapter"]["adapter"]
            for k, img in worker.refs(name)[_key(folder)].items():
                assert compare.digest(img) == entry["images"][k]


def _key(folder: str) -> str:
    return next(k for k, v in worker.REF_SETS.items() if v == folder)


def test_the_stand_in_serves_ranges_and_only_its_files():
    server = scene.Server("r2_400k")
    with server(scene.SPLAT_URL, {"Range": "bytes=10-"}) as part:
        assert part.status == 206
        assert part.getheader("Content-Range") == f"bytes 10-22400361/{22400362}"
        assert part.read() == scene.splat("r2_400k")[10:]
    with server("/index.html") as page:
        assert b"/assets/main-" in page.read()
    for path in ("/assets/../../README.md", "/__key", "/assets/missing.js"):
        with pytest.raises(FileNotFoundError):
            server(path)


def test_the_lossless_page_encodes_png_in_place_of_jpeg():
    name = next(n for n in run.page_manifest()["files"] if "main-" in n)
    plain = scene.Server("r2_400k")("/" + name).read()
    png = scene.Server("r2_400k", lossless=True)("/" + name).read()
    assert scene.PNG_ENCODE[0] in plain and scene.PNG_ENCODE[0] not in png
    assert png.count(scene.PNG_ENCODE[1]) == 1
    assert png.replace(scene.PNG_ENCODE[1], scene.PNG_ENCODE[0]) == plain


# --------------------------------------------------------------- the numbers
def test_compare_gives_the_gate_numbers():
    ref = {"a": np.full((2, 2, 3), 100, np.uint8), "b": np.full((2, 2, 3), 50, np.uint8)}
    test = {k: v.copy() for k, v in ref.items()}
    test["a"][0, 0] = [103, 100, 100]  # one pixel off by 3 in one channel
    test["b"][1, 1] = [70, 50, 50]  # one pixel off by 20
    got = compare.compare(ref, test)
    assert got["images"] == 2 and got["max"] == 20
    assert got["mean"] == round(23 / 24, 4) and got["signed_mean"] == round(23 / 24, 4)
    assert got["worst_image"] == round(20 / 12, 4)
    assert got["frac_gt2"] == round(2 / 8, 5) and got["frac_gt16"] == round(1 / 8, 5)
    assert compare.compare(ref, {"a": ref["a"]})["missing"] == ["b"]


@pytest.mark.parametrize(
    "text, kind",
    [
        ("Intel(R) UHD Graphics (TGL GT1)", "intel"),
        ("ANGLE (Intel, Mesa Intel(R) UHD Graphics (TGL GT1), OpenGL ES 3.2)", "intel"),
        ("ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)", "intel"),
        ("Apple M4 Pro", "apple"),
        ("ANGLE (Apple, ANGLE Metal Renderer: Apple M4 Pro, Unspecified Version)", "apple"),
        ("NVIDIA GeForce RTX 4090", "nvidia"),
        ("ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 4090/PCIe/SSE2, OpenGL ES 3.2)", "nvidia"),
        ("llvmpipe (LLVM 15.0.7, 256 bits)", "software"),
        ("SwiftShader Device (Subzero)", "software"),
        (
            "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader)",
            "software",
        ),
        ("Microsoft Basic Render Driver", "software"),
        ("AMD Radeon(TM) Graphics", "amd"),
        (None, None),
    ],
)
def test_adapters_are_told_apart_by_name(text, kind):
    assert run.gpu_class(text) == kind


def test_each_round_turns_the_order():
    keys = ["d", "chrome", "d_rgba8"]
    orders = [run.round_order(keys, r) for r in (1, 2, 3, 4)]
    assert orders[0] == keys and orders[3] == keys
    assert {o[0] for o in orders[:3]} == set(keys), "each path goes first once"


def _record(key, rnd, adapter, rate, room_d, room_spark, fog_spark=0.4, images="x"):
    quality = {
        "room200k": {
            "images": {"s000_wrist": images},
            "vs_d_4090": {"mean": room_d, "worst_image": room_d, "max": 9, "frac_gt2": 0.1},
            "vs_spark_4090": {
                "mean": room_spark,
                "worst_image": room_spark,
                "max": 9,
                "frac_gt2": 0.1,
            },
        },
        "r2_400k": {
            "images": {"s000_wrist": "f"},
            "vs_spark_4090": {
                "mean": fog_spark,
                "worst_image": fog_spark,
                "max": 1,
                "frac_gt2": 0.0,
            },
        },
    }
    return {
        "key": key,
        "path": "d" if key.startswith("d") else "chrome",
        "round": rnd,
        "ok": True,
        "adapter": {"adapter": adapter, "adapter_type": "IntegratedGPU", "backend": "Vulkan"},
        "timed": {"steps_per_s": rate, "frame_ms": {"median": 1e3 / rate}},
        "quality": quality,
        "warnings": [],
    }


INTEL_D, INTEL_CHROME = "Intel(R) UHD Graphics", "ANGLE (Intel, Mesa Intel(R) UHD Graphics)"


def test_the_checks_follow_the_readme():
    runs = [
        _record("d", 1, INTEL_D, 40.0, 0.3, 1.8),
        _record("d", 2, INTEL_D, 44.0, 0.3, 1.8),
        _record("d", 3, INTEL_D, 42.0, 0.3, 1.8),
        _record("chrome", 1, INTEL_CHROME, 10.0, 3.0, 2.7),
        _record("chrome", 2, INTEL_CHROME, 12.0, 3.0, 2.7),
        _record("chrome", 3, INTEL_CHROME, 11.0, 3.0, 2.7),
    ]
    s = run.summarize(runs)
    assert s["pass"] is True and s["same_gpu"] == {
        "d": ["intel"],
        "chrome": ["intel"],
        "same": True,
    }
    assert s["d_over_chrome"] == {
        "median_ratio": round(42 / 11, 3),
        "per_round": [4.0, 3.667, 3.818],
    }
    assert s["paths"]["d"]["steps_per_s"]["median"] == 42.0
    assert s["paths"]["d"]["quality"]["room200k"]["identical_across_runs"] is True
    assert [c["pass"] for c in s["checks"]] == [True, True, True, True]


def test_the_worst_run_decides_and_a_slower_d_fails():
    runs = [
        _record("d", 1, INTEL_D, 9.0, 0.3, 1.8),
        _record("d", 2, INTEL_D, 9.0, 1.6, 3.2, images="y"),  # one run off
        _record("chrome", 1, INTEL_CHROME, 10.0, 3.0, 2.7),
        _record("chrome", 2, INTEL_CHROME, 10.0, 3.0, 2.7),
    ]
    s = run.summarize(runs)
    by = {c["name"].split(":")[0] + str(c["limit"]): c for c in s["checks"]}
    assert by["room200k1.5"]["value"] == 1.6 and by["room200k1.5"]["pass"] is False
    assert by["room200k3.0"]["value"] == 3.2 and by["room200k3.0"]["pass"] is False
    assert by["r2_400k1.0"]["value"] == 0.9 and by["r2_400k1.0"]["pass"] is False
    assert s["pass"] is False
    assert s["paths"]["d"]["quality"]["room200k"]["identical_across_runs"] is False


def test_paths_on_different_gpus_are_flagged():
    runs = [
        _record("d", 1, INTEL_D, 40.0, 0.3, 1.8),
        _record(
            "chrome",
            1,
            "ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 3060 Laptop GPU)",
            30.0,
            3,
            2.7,
        ),
    ]
    s = run.summarize(runs)
    assert s["same_gpu"]["same"] is False
    speed = s["checks"][-1]
    assert speed["same_gpu"] is False and "same GPU" in speed["name"]


def test_a_failed_run_is_kept_with_its_error():
    failed = {"key": "d", "path": "d", "round": 1, "ok": False, "error": {"type": "X"}}
    s = run.summarize([failed])
    assert (s["paths"]["d"]["runs"], s["paths"]["d"]["ok"]) == (1, 0)
    assert all(c["pass"] is None for c in s["checks"]) and s["pass"] is True


# ------------------------------------------------ renderer D on this machine
@pytest.fixture
def adapter():
    """Renderer D can draw here (else skip, or fail with WE_TEST_REQUIRE_WGSL=1)."""
    from world_engine_client import render_wgsl

    if not REQUIRED and not render_wgsl.available():
        pytest.skip("wgpu is not installed")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            render_wgsl.find_adapter()
    except render_wgsl.NoRenderAdapter as exc:
        if REQUIRED:
            pytest.fail(f"WE_TEST_REQUIRE_WGSL=1 but no adapter renders: {exc}")
        pytest.skip(f"no adapter renders here: {exc}")


def test_renderer_d_here_draws_what_it_drew_on_the_rtx_4090(adapter, tmp_path):
    """A short run of the bench's worker (three timed states): its images of
    both scenes are within README.md's limits of the RTX 4090 references, on
    whatever adapter this machine has (CI: lavapipe, the bundled SwiftShader)."""
    spec = {
        "path": "d",
        "label": "test",
        "adapter": os.environ.get("WE_RENDER_ADAPTER") or "auto",
        "steps": 3,
        "min_steps": 3,
        "max_seconds": 0,
        "warmup_seconds": 0,
        "cache_dir": str(tmp_path / "cache"),
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    done = subprocess.run(
        [
            sys.executable,
            str(BENCH / "worker.py"),
            str(tmp_path / "spec.json"),
            str(tmp_path / "out.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    record = json.loads((tmp_path / "out.json").read_text())
    assert record["ok"], (record.get("error"), done.stderr[-2000:])
    assert record["timed"]["frames"] == 3
    room = record["quality"]["room200k"]
    assert room["vs_d_4090"]["mean"] <= run.THRESHOLDS["d_vs_d_4090"], room["vs_d_4090"]
    assert room["vs_spark_4090"]["mean"] <= run.THRESHOLDS["d_vs_spark_4090"], room["vs_spark_4090"]
    assert record["quality"]["r2_400k"]["vs_spark_4090"]["mean"] <= 3.0
