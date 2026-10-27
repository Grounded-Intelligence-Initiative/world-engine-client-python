"""Package metadata: the version, the Python floor, the classifiers and the CI
matrix agree."""

import re
import tomllib
from pathlib import Path

import world_engine_client

ROOT = Path(__file__).resolve().parents[1]


def test_metadata_agrees_with_itself():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == world_engine_client.__version__
    assert project["requires-python"] == ">=3.11"
    prefix = "Programming Language :: Python :: 3."
    listed = sorted(
        c.removeprefix("Programming Language :: Python :: ")
        for c in project["classifiers"]
        if c.startswith(prefix)
    )
    assert listed and all(tuple(map(int, v.split("."))) >= (3, 11) for v in listed)
    workflow = (ROOT / ".github/workflows/test.yml").read_text()
    matrices = re.findall(r"python: \[(.*?)\]", workflow)
    assert matrices, "the CI workflow has a Python matrix"
    for matrix in matrices:
        assert sorted(v.strip().strip('"') for v in matrix.split(",")) == listed
