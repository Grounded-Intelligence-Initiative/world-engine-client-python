"""The integrated-GPU bench (bench/igpu): its scenes are the same bytes on every
machine, its references are the ones it says, the engine's capture page comes
from outside the repository and is checked, its numbers and checks follow
README.md, and renderer D here draws what it drew on the RTX 4090 (that one
needs wgpu and an adapter; it is skipped without one). The real capture page
is used where IGPU_BENCH_CAPTURE_PAGE names it; elsewhere a stand-in."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench" / "igpu"
if not BENCH.is_dir():  # an sdist: the bench is the repository's only
    pytest.skip("bench/igpu is not here", allow_module_level=True)
sys.path.insert(0, str(BENCH))

import compare  # noqa: E402
import pack_capture_page  # noqa: E402
import page  # noqa: E402
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


def fake_page(root: Path) -> dict:
    """A stand-in capture page build in ``root``, and its manifest."""
    files = {
        "index.html": b'<script type="module" src="/assets/main-T3st.js"></script>',
        "assets/main-T3st.js": b"let b=await c." + scene.PNG_ENCODE[0] + b";send(b)",
        "assets/scene-T3st.js": b"export const splats = 1;",
    }
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    return {"engine_commit": "f00d" * 10, "build_stamp": {"mode": "production"}, "files": digests}


def test_the_manifest_names_one_production_build():
    m = page.manifest()
    assert set(m["files"]) == {
        "index.html",
        "assets/main-C0ZK_nJr.js",
        "assets/sceneEnvironment-BecPG80m.js",
    }
    assert m["build_stamp"]["mode"] == "production" and len(m["engine_commit"]) == 40
    assert page.archive_name(m) == "igpu-capture-page-1af5543.zip"


def test_the_engine_build_is_not_in_the_repository():
    """The capture page is the build of the engine, whose repository is
    private: only its manifest and notes are committed, and git ignores the
    build where page.py would find it."""
    try:
        done = subprocess.run(
            ["git", "-C", str(BENCH), "ls-files", "capture_page"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except OSError:
        pytest.skip("no git here")
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = set(done.stdout.split())
    assert tracked <= {
        "capture_page/.gitattributes",
        "capture_page/LICENSES.md",
        "capture_page/MANIFEST.json",
    }, tracked
    assert "capture_page/MANIFEST.json" in tracked
    for name in ("capture_page/index.html", "capture_page/assets/main-C0ZK_nJr.js"):
        ignored = subprocess.run(["git", "-C", str(BENCH), "check-ignore", "-q", name])
        assert ignored.returncode == 0, f"git does not ignore {name}"


def test_a_capture_page_is_taken_from_a_directory_or_a_zip_and_checked(tmp_path):
    src = tmp_path / "dist"
    m = fake_page(src)
    got, record = page.load(src, tmp_path / "unused", m)
    assert got == src and record["kind"] == "directory" and record["files"] == m["files"]
    assert not (tmp_path / "unused").exists(), "a directory is used where it is"
    for prefix in ("", "capture_page/"):  # the zip of pack_capture_page, or of a folder
        archive = tmp_path / f"page{len(prefix)}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name in m["files"]:
                z.write(src / name, prefix + name)
            z.writestr("../outside.js", b"not a manifest file")
        dest = tmp_path / f"out{len(prefix)}"
        got, record = page.load(archive, dest, m)
        assert got == dest and record["kind"] == "zip"
        assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == (
            sorted(m["files"])
        ), "only the manifest's files, under its names"
        assert not (tmp_path / "outside.js").exists()
    (src / "assets" / "scene-T3st.js").write_bytes(b"export const splats = 2;")
    with pytest.raises(SystemExit, match="scene-T3st.js"):
        page.load(src, tmp_path / "x", m)
    with pytest.raises(SystemExit, match="neither a directory nor a zip"):
        page.load(tmp_path / "missing.zip", tmp_path / "x", m)


def test_the_zip_packs_the_same_bytes_and_the_bench_takes_it(tmp_path):
    m = fake_page(tmp_path / "dist")
    notes = tmp_path / "LICENSES.md"
    notes.write_text("notes")
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    digest = pack_capture_page.pack(tmp_path / "dist", a, m, extra={"LICENSES.md": notes})
    assert pack_capture_page.pack(tmp_path / "dist", b, m, extra={"LICENSES.md": notes}) == digest
    assert a.read_bytes() == b.read_bytes()
    got, record = page.load(a, tmp_path / "page", m)
    assert record["kind"] == "zip" and (got / "assets" / "main-T3st.js").is_file()
    assert "LICENSES.md" in zipfile.ZipFile(a).namelist()


def test_without_a_capture_page_the_bench_says_where_to_get_one(monkeypatch, capsys):
    monkeypatch.delenv(page.ENV, raising=False)
    monkeypatch.setattr(page, "DEFAULT", Path(tempfile.gettempdir()) / "no-page-here")
    with pytest.raises(SystemExit) as stop:
        run.main(["--paths", "d,chrome", "--rounds", "1"])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "--capture-page" in err and "igpu-capture-page-1af5543.zip" in err
    assert "pack_capture_page.py" in err and "--paths d" in err


@pytest.fixture
def real_page(tmp_path):
    """The engine's capture page where this machine has it (IGPU_BENCH_CAPTURE_PAGE),
    checked against the manifest."""
    where = page.locate()
    if where is None:
        pytest.skip(f"no capture page here ({page.ENV} names none)")
    return page.load(where, tmp_path / "real-page")


def test_the_capture_page_here_is_the_one_its_manifest_names(real_page):
    page_dir, record = real_page
    html = (page_dir / "index.html").read_text()
    for name in record["files"]:
        if name.startswith("assets/"):
            assert f'"/{name}"' in html
    main = next(n for n in record["files"] if "main-" in n)
    plain = scene.Server("r2_400k", page_dir=page_dir)("/" + main).read()
    png = scene.Server("r2_400k", page_dir=page_dir, lossless=True)("/" + main).read()
    assert plain.count(scene.PNG_ENCODE[0]) == 1 and png.count(scene.PNG_ENCODE[1]) == 1


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


def test_the_stand_in_serves_ranges_and_only_its_files(tmp_path):
    fake_page(tmp_path / "page")
    (tmp_path / "secret.txt").write_text("not the page's")
    server = scene.Server("r2_400k", page_dir=tmp_path / "page")
    with server(scene.SPLAT_URL, {"Range": "bytes=10-"}) as part:
        assert part.status == 206
        assert part.getheader("Content-Range") == f"bytes 10-22400361/{22400362}"
        assert part.read() == scene.splat("r2_400k")[10:]
    with server("/index.html") as index:
        assert b"/assets/main-" in index.read()
    for path in ("/assets/../../secret.txt", "/__key", "/assets/missing.js"):
        with pytest.raises(FileNotFoundError):
            server(path)


def test_the_lossless_page_encodes_png_in_place_of_jpeg(tmp_path):
    fake_page(tmp_path / "page")
    name = "/assets/main-T3st.js"
    plain = scene.Server("r2_400k", page_dir=tmp_path / "page")(name).read()
    png = scene.Server("r2_400k", page_dir=tmp_path / "page", lossless=True)(name).read()
    assert scene.PNG_ENCODE[0] in plain and scene.PNG_ENCODE[0] not in png
    assert png.count(scene.PNG_ENCODE[1]) == 1
    assert png.replace(scene.PNG_ENCODE[1], scene.PNG_ENCODE[0]) == plain
    twice = tmp_path / "page" / "assets" / "main-Tw1ce.js"
    twice.write_bytes(scene.PNG_ENCODE[0] * 2)
    with pytest.raises(SystemExit, match="not there once"):
        scene.Server("r2_400k", page_dir=tmp_path / "page", lossless=True)("/assets/main-Tw1ce.js")


def test_a_run_leaves_no_temporary_directory_unless_asked(tmp_path, monkeypatch):
    """The render cache (the scenes' plaintext PLYs, the page's files, about
    80 MB) and a capture page taken from a zip go into one temporary
    directory, removed at the end; --keep-cache keeps it."""
    monkeypatch.setattr(run.machine, "collect", lambda: {"gpus": {}})
    caches = []

    def fake_worker(spec, env, timeout):
        caches.append(Path(spec["cache_dir"]))
        caches[-1].mkdir(parents=True, exist_ok=True)
        (caches[-1] / "scene.ply").write_bytes(b"plaintext")
        return {"path": spec["path"], "label": spec["label"], "ok": False, "error": {"type": "X"}}

    monkeypatch.setattr(run, "run_worker", fake_worker)
    argv = ["--paths", "d", "--rounds", "1", "--no-images", "--out", str(tmp_path / "r.json")]
    assert run.main(argv) == 0
    assert len(caches) == 1 and not caches[0].exists(), "the render cache is gone"
    assert not list(Path(tempfile.gettempdir()).glob("igpu-bench-*")), "and its directory"
    assert run.main([*argv, "--keep-cache"]) == 0
    assert caches[1].exists(), "--keep-cache keeps it"


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
    assert s["pass"] is True and s["complete"] is True and s["verdict"] == "PASS"
    assert s["same_gpu"] == {
        "d": ["intel uhd graphics"],
        "chrome": ["intel uhd graphics"],
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


NVIDIA_D = "NVIDIA GeForce RTX 3060 Laptop GPU"
NVIDIA_CHROME = (
    "ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 3060 Laptop GPU/PCIe/SSE2, OpenGL ES 3.2)"
)


def _failed(key, rnd):
    path = "d" if key.startswith("d") else "chrome"
    return {"key": key, "path": path, "round": rnd, "ok": False, "error": {"type": "X"}}


def _rounds(key, adapter, rate, n=3):
    return [
        _record(key, r, adapter, rate, 0.3 if key == "d" else 3.0, 1.8) for r in range(1, n + 1)
    ]


def test_paths_on_different_gpus_are_flagged_and_their_speed_not_judged():
    """A laptop with an Intel and an NVIDIA GPU: renderer D on one, Chrome on
    the other. Their ratio says nothing about either GPU, whichever is faster."""
    for d, c, rate_d, rate_c in (
        (NVIDIA_D, INTEL_CHROME, 60.0, 8.0),  # D on the discrete GPU: looked like a pass
        (INTEL_D, NVIDIA_CHROME, 8.0, 12.0),  # D on the integrated GPU: looked like a fail
    ):
        s = run.summarize(_rounds("d", d, rate_d) + _rounds("chrome", c, rate_c))
        speed = s["checks"][-1]
        assert s["same_gpu"]["same"] is False and speed["same_gpu"] is False
        assert speed["value"] == round(rate_d / rate_c, 3) and speed["pass"] is None
        assert "different GPUs" in speed["name"] and "not judged" in speed["name"]
        assert s["pass"] is None and s["complete"] is False and s["verdict"] == "INCOMPLETE"


def test_a_failed_run_is_kept_with_its_error():
    s = run.summarize([_failed("d", 1)])
    assert (s["paths"]["d"]["runs"], s["paths"]["d"]["ok"]) == (1, 0)
    assert all(c["pass"] is None for c in s["checks"])
    assert s["pass"] is None and s["complete"] is False and s["failed_runs"] == ["d round 1"]


def test_without_a_browser_path_run_the_speed_is_not_judged():
    s = run.summarize(_rounds("d", INTEL_D, 40.0) + [_failed("chrome", r) for r in (1, 2, 3)])
    assert [c["pass"] for c in s["checks"]] == [True, True, True, None]
    assert s["pass"] is None and len(s["failed_runs"]) == 3
    assert any("not judged" in why for why in s["incomplete_because"])


def test_runs_that_failed_in_some_rounds_leave_the_result_incomplete():
    runs = [_record("d", 1, INTEL_D, 40.0, 0.3, 1.8), _failed("d", 2), _failed("d", 3)]
    s = run.summarize(runs + _rounds("chrome", INTEL_CHROME, 10.0))
    assert [c["pass"] for c in s["checks"]] == [True, True, True, True]
    assert s["pass"] is None and s["failed_runs"] == ["d round 2", "d round 3"]


def test_planned_runs_that_never_ran_leave_the_result_incomplete():
    runs = _rounds("d", INTEL_D, 40.0) + _rounds("chrome", INTEL_CHROME, 10.0)
    planned = [(k, r) for r in (1, 2, 3) for k in ("d", "chrome")] + [("chrome_png", 0)]
    s = run.summarize(runs, planned)
    assert s["missing_runs"] == ["chrome_png round 0"] and s["pass"] is None
    assert run.summarize(runs, planned[:-1])["pass"] is True


def test_a_failed_check_fails_even_when_runs_are_missing():
    runs = [_record("d", 1, INTEL_D, 40.0, 2.0, 1.8), _failed("chrome", 1)]
    s = run.summarize(runs)
    assert s["checks"][0]["pass"] is False and s["pass"] is False and s["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "d, chrome, same",
    [
        (
            "NVIDIA GeForce RTX 4090",
            "ANGLE (NVIDIA Corporation, NVIDIA GeForce RTX 4090/PCIe/SSE2, OpenGL ES 3.2)",
            True,
        ),
        (
            "Intel(R) UHD Graphics (TGL GT1)",
            "ANGLE (Intel, Mesa Intel(R) UHD Graphics (TGL GT1), OpenGL ES 3.2)",
            True,
        ),
        (
            "Intel(R) UHD Graphics",
            "ANGLE (Intel, Intel(R) UHD Graphics (0x00009A60) Direct3D11 vs_5_0 ps_5_0, D3D11)",
            True,
        ),
        (
            "Intel(R) UHD Graphics (TGL GT1)",
            "ANGLE (Intel, Vulkan 1.3.255 (Intel(R) UHD Graphics (TGL GT1) (0x00009A60)), "
            "Intel open-source Mesa driver-23.2.1)",
            True,
        ),
        (
            "Apple M4 Pro",
            "ANGLE (Apple, ANGLE Metal Renderer: Apple M4 Pro, Unspecified Version)",
            True,
        ),
        (
            "AMD Radeon(TM) Graphics",
            "ANGLE (AMD, AMD Radeon(TM) Graphics (0x00001638) Direct3D11 vs_5_0 ps_5_0, D3D11)",
            True,
        ),
        (
            "llvmpipe (LLVM 15.0.7, 256 bits)",
            "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), "
            "SwiftShader driver)",
            True,
        ),
        (
            "AMD Radeon(TM) Graphics",
            "ANGLE (AMD, AMD Radeon RX 6800M (0x000073DF) Direct3D11 vs_5_0 ps_5_0, D3D11)",
            False,
        ),
        (
            "Intel(R) UHD Graphics",
            "ANGLE (Intel, Intel(R) Arc(TM) A370M Graphics (0x00005693) Direct3D11 vs_5_0 "
            "ps_5_0, D3D11)",
            False,
        ),
        ("Intel(R) UHD Graphics", NVIDIA_CHROME, False),
        ("llvmpipe (LLVM 15.0.7, 256 bits)", NVIDIA_CHROME, False),
    ],
)
def test_the_two_paths_are_on_one_gpu_by_its_name(d, chrome, same):
    """By the device's name, not its vendor: an AMD or Intel laptop can have
    an integrated and a discrete GPU of one vendor. Any CPU renderer on both
    sides is the one CPU."""
    s = run.summarize(
        [_record("d", 1, d, 40.0, 0.3, 1.8), _record("chrome", 1, chrome, 9.0, 3, 2.7)]
    )
    assert s["same_gpu"]["same"] is same


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
