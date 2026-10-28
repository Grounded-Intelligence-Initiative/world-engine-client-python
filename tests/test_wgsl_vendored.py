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
