#!/usr/bin/env python
"""Copy renderer D from the engine into this package, or check the copy.

The engine repository is the one source of renderer D (design doc 16 §9.3):
its shaders in ``frontend/src/render/wgsl/`` and its Python host core in
``world_engine/renderer/wgsl/``. world-engine-client ships a copy of both in
the package ``world_engine_client.wgsl`` (``src/world_engine_client/wgsl/``), so
the wheel carries the shaders it runs and never runs a shader that a server
sends. This script makes that copy and records what it copied in
``wgsl/SOURCE.json``: the pipeline version, the engine commit and the SHA-256
of every file. ``tests/test_wgsl_vendored.py`` checks the copy against
SOURCE.json in every test run; ``--check`` compares it with an engine checkout,
byte for byte, before a release. The engine's tests do the same when they find
this repository next to the engine (world-engine-dev).

    python scripts/sync_wgsl.py ../world-engine            # copy, write SOURCE.json
    python scripts/sync_wgsl.py ../world-engine --check    # exit 1 when they differ

Do not edit the copied files here: change them in the engine, then copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
VENDORED = HERE / "src" / "world_engine_client" / "wgsl"
# engine directory -> which of its files are copied
SOURCES = {
    Path("frontend") / "src" / "render" / "wgsl": ("*.wgsl", "pipeline.json", "README.md"),
    Path("world_engine") / "renderer" / "wgsl": ("*.py",),
}
RECORD = "SOURCE.json"


def engine_files(engine: Path) -> dict[str, tuple[str, bytes]]:
    """name -> (path in the engine, bytes) of every file the copy takes."""
    out: dict[str, tuple[str, bytes]] = {}
    for rel, patterns in SOURCES.items():
        for pattern in patterns:
            for p in sorted((engine / rel).glob(pattern)):
                if p.is_file():
                    if p.name in out:
                        raise SystemExit(f"two engine files are named {p.name}")
                    out[p.name] = (str(rel / p.name), p.read_bytes())
    return out


def copied_files() -> dict[str, bytes]:
    if not VENDORED.is_dir():
        return {}
    return {
        p.name: p.read_bytes()
        for p in sorted(VENDORED.iterdir())
        if p.is_file() and p.name != RECORD and not p.name.startswith(".")
    }


def engine_commit(engine: Path) -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", str(engine), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(engine), "status", "--porcelain", "--", *map(str, SOURCES)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def differences(theirs: dict[str, bytes], ours: dict[str, bytes]) -> list[str]:
    out = []
    for name in sorted(set(theirs) | set(ours)):
        if name not in ours:
            out.append(f"only in the engine: {name}")
        elif name not in theirs:
            out.append(f"only in the copy: {name}")
        elif theirs[name] != ours[name]:
            out.append(f"differs: {name}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("engine", type=Path, help="a world-engine checkout (the repository root)")
    ap.add_argument("--check", action="store_true", help="compare only; exit 1 on a difference")
    a = ap.parse_args(argv)
    if not (a.engine / "frontend" / "src" / "render" / "wgsl" / "pipeline.json").is_file():
        print(f"{a.engine}: no frontend/src/render/wgsl/pipeline.json", file=sys.stderr)
        return 2
    theirs = engine_files(a.engine)
    if a.check:
        diff = differences({k: v[1] for k, v in theirs.items()}, copied_files())
        if diff:
            print("the copy of renderer D is not the engine's:", file=sys.stderr)
            for line in diff:
                print("  " + line, file=sys.stderr)
            print(f"run: python scripts/{Path(__file__).name} {a.engine}", file=sys.stderr)
            return 1
        print(f"the copy of renderer D matches {a.engine} ({len(theirs)} files)")
        return 0
    VENDORED.mkdir(parents=True, exist_ok=True)
    for old in VENDORED.iterdir():
        if old.is_file() and old.name not in theirs:
            old.unlink()
    for name, (_, data) in theirs.items():
        (VENDORED / name).write_bytes(data)
    pipeline = json.loads(theirs["pipeline.json"][1])["pipeline"]
    record = {
        "comment": (
            "Copied by scripts/sync_wgsl.py from the engine repository; change these files "
            "there, not here."
        ),
        "pipeline": pipeline,
        "engine_commit": engine_commit(a.engine),
        "files": {
            name: {"source": src, "sha256": hashlib.sha256(data).hexdigest()}
            for name, (src, data) in sorted(theirs.items())
        },
    }
    (VENDORED / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    print(f"copied {len(theirs)} files of pipeline {pipeline} from {a.engine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
