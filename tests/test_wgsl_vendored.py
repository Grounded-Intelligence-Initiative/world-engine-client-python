"""The copy of renderer D in this package (world_engine_client.wgsl): the
engine's shaders and host core, copied by scripts/sync_wgsl.py, recorded in
wgsl/SOURCE.json. The copy must be what the record says (nobody edits it
here), and the host must speak the pipeline version the shaders declare.
With WE_ENGINE_SRC (a world-engine checkout) the copy is also compared with the
engine, byte for byte."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import world_engine_client
from world_engine_client import render_wgsl

ROOT = Path(__file__).resolve().parents[1]
COPY = Path(world_engine_client.__file__).resolve().parent / "wgsl"


def test_the_copy_is_what_its_record_says():
    record = json.loads((COPY / "SOURCE.json").read_text())
    files = {p.name for p in COPY.iterdir() if p.is_file() and p.name != "SOURCE.json"}
    files.discard("__pycache__")
    assert files == set(record["files"]), "files added or removed by hand: run scripts/sync_wgsl.py"
    for name, entry in record["files"].items():
        digest = hashlib.sha256((COPY / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], (
            f"{name} was edited here; change it in the engine and copy"
        )
    assert record["engine_commit"] and "+dirty" not in record["engine_commit"], (
        "copied from an engine tree with uncommitted changes"
    )


def test_the_host_speaks_the_pipeline_of_its_shaders():
    manifest = json.loads((COPY / "pipeline.json").read_text())
    record = json.loads((COPY / "SOURCE.json").read_text())
    assert manifest["pipeline"] == record["pipeline"] == render_wgsl.PIPELINE
    for module, files in manifest["modules"].items():
        for f in files:
            assert (COPY / f).is_file(), f"module {module} needs {f}"
    from world_engine_client.wgsl import render_wgsl as core

    assert core.PIPELINE == render_wgsl.PIPELINE
    assert render_wgsl.WGSL_DIR == COPY, "the client draws with the shaders it ships"


def test_the_wheel_takes_the_copy():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel = project["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/world_engine_client"]
    assert "exclude" not in wheel and "only-include" not in wheel
    ignored = (ROOT / ".gitignore").read_text()
    assert "wgsl" not in ignored and ".wgsl" not in ignored


@pytest.mark.skipif(
    not os.environ.get("WE_ENGINE_SRC"), reason="no engine checkout (WE_ENGINE_SRC)"
)
def test_the_copy_matches_the_engine():
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sync_wgsl.py"),
            os.environ["WE_ENGINE_SRC"],
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(
    not os.environ.get("WE_ENGINE_SRC"), reason="no engine checkout (WE_ENGINE_SRC)"
)
def test_the_test_scenes_wem1_is_what_the_engine_compiles():
    """The tests' WEM1 box (tests/d_scene.py) is the engine converter's output
    for the same GLB, byte for byte, so the tests read the format servers send."""
    import importlib.util

    import d_scene

    path = Path(os.environ["WE_ENGINE_SRC"]) / "world_engine" / "assets" / "wem1.py"
    spec = importlib.util.spec_from_file_location("_engine_wem1", path)
    wem1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wem1)
    assert wem1.glb_to_wem1(d_scene.box_glb()) == d_scene.box_wem1()


# What this package ships as pipeline wgsl/1, from 0.7.0 on: the SHA-256 of
# pipeline.json and of each .wgsl file (the engine's tests/test_render_protocol.py
# keeps the same record). Engines draw scenes for wgsl/1 clients by what that
# version names, so a copy that draws differently must come with a new pipeline
# version; change this record only for a version no release has shipped.
SHIPPED = {
    "wgsl/1": {
        "pipeline.json": "84675474229983334884ad838bb77df8675e356706e5ec1308ca61099558ad5a",
        "common.wgsl": "ff5616bf5b8f7859c5a443309c3b651e06d39810e39add341dc4dbf473f80a44",
        "cubeuv.wgsl": "08b02f30a30756e46a673e9a18dd1f3ba09130fa8efa1d3a974bb3f62effe7a7",
        "material.wgsl": "9e900abef351a62bb58b4d0db6a812d07ec460ec508ee78d3003894d1b72df63",
        "mesh.wgsl": "218cab3d9351d67d374539367e5e4a9301462bd3012798f62f7647b3cec48be1",
        "pmrem.wgsl": "1beebbdfda25b39fb7044bb672f3d3d8d906ae856306a4546e35aca50f5aa44b",
        "preprocess.wgsl": "2dbbdd1f9e6dfedaa9da9dfbc95e9c755987296b0d5d0cfe050c738782068644",
        "resolve.wgsl": "398064c223b93530d1556bf1928ddbd814b2ea73d6a94c56574ec567ecdda30c",
        "shadow.wgsl": "6d2a494c5c236ddd45a5da4294f2ece1b34265d05a1d9c676a2c9c7e76f37acc",
        "sort.wgsl": "eda70bf818910fed36bc414b623190add93dc4757e363de8bb7e8df68762e551",
        "splat.wgsl": "8abd95f2e2e3f8ba1058a18feca323fb20d873ca454122bbc357ba2e8fc58bde",
    }
}


def test_a_pipeline_version_names_one_drawing():
    record = json.loads((COPY / "SOURCE.json").read_text())
    version = record["pipeline"]
    if version in SHIPPED:
        now = {
            name: entry["sha256"]
            for name, entry in record["files"].items()
            if name.endswith(".wgsl") or name == "pipeline.json"
        }
        assert now == SHIPPED[version], (
            f"the shaders of pipeline {version} changed, and world-engine-client 0.7.0 "
            "ships them as they were: copy a drawing that differs under a new version"
        )
