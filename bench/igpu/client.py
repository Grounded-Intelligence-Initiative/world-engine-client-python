"""Which world-engine-client the bench runs, and whether it is the one of
this checkout.

refs/ holds what renderer D of this checkout drew on an RTX 4090, and CI
checks that this checkout's renderer D still draws within the README's
limits of them. A bench run means something only for that renderer D, so the
bench compares the installed package with this checkout's
src/world_engine_client, file by file, and refs/refs.json names the renderer
D it was made with (``made_with.renderer_d``, a digest of the files that draw
D's images).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKOUT_SRC = ROOT / "src" / "world_engine_client"
# The files that decide renderer D's pixels: its host, the asset reader, and
# the vendored pipeline, less its docs, its source list and the adapter search.
RENDERER_D = ("render_wgsl.py", "assets.py", "wgsl/")
NOT_RENDERER_D = ("wgsl/README.md", "wgsl/SOURCE.json", "wgsl/adapters.py", "wgsl/_probe.py")


def package_files(root: Path) -> dict[str, str]:
    """{path relative to the package: SHA-256} of every file of a package
    directory, bytecode left out."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"):
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def renderer_d_digest(files: dict[str, str]) -> str:
    """One digest of the renderer D files among ``files`` (package_files)."""
    chosen = {
        k: v for k, v in files.items() if k.startswith(RENDERER_D) and k not in NOT_RENDERER_D
    }
    text = "".join(f"{k} {v}\n" for k, v in sorted(chosen.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def checkout_version(root: Path = ROOT) -> str | None:
    try:
        text = (root / "pyproject.toml").read_text()
    except OSError:
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def git_describe(root: Path = ROOT) -> str | None:
    """``git describe`` of the checkout (a tag, commits since it, the commit,
    -dirty), or None without git."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def info(refs_made_with: dict | None = None, checkout_src: Path = CHECKOUT_SRC) -> dict:
    """The installed client against this checkout and against refs/.

    ``matches_checkout``: every file of the imported package is the same as
    in the checkout's src/world_engine_client (None when the bench is not in a
    checkout). ``refs_made_with_this_renderer_d``: refs.json names this
    renderer D (None when it names none)."""
    import world_engine_client

    installed = Path(world_engine_client.__file__).resolve().parent
    files = package_files(installed)
    out: dict = {
        "installed": {
            "version": getattr(world_engine_client, "__version__", None),
            "dir": str(installed),
        },
        "renderer_d": renderer_d_digest(files),
        "checkout": None,
        "matches_checkout": None,
        "differs": [],
    }
    if (checkout_src / "__init__.py").is_file():
        theirs = package_files(checkout_src)
        differs = sorted(k for k in set(files) | set(theirs) if files.get(k) != theirs.get(k))
        out["checkout"] = {
            "version": checkout_version(checkout_src.parents[1]),
            "git": git_describe(checkout_src.parents[1]),
            "dir": str(checkout_src.resolve()),
        }
        out["matches_checkout"] = not differs
        out["differs"] = differs[:20]
    made = (refs_made_with or {}).get("renderer_d")
    out["refs_renderer_d"] = made
    out["refs_made_with_this_renderer_d"] = None if made is None else made == out["renderer_d"]
    return out


def describe(c: dict) -> list[str]:
    """Lines for the terminal: which client runs, and what is off."""
    inst = c["installed"]
    lines = [f"world-engine-client {inst['version']} from {inst['dir']}"]
    if c["matches_checkout"] is True:
        lines[0] += f": the files of this checkout ({c['checkout']['git'] or 'no git'})"
    elif c["matches_checkout"] is False:
        co = c["checkout"]
        lines.append(
            f"WARNING: the installed world-engine-client is not this checkout's "
            f"({co['version']}, {co['git'] or 'no git'}): {len(c['differs'])} file(s) differ, "
            f"e.g. {', '.join(c['differs'][:4])}. refs/ goes with this checkout: install it "
            "from here (pip install '.[render]') or check out the tag of the installed "
            f"version (git checkout v{inst['version']}). The summary will be incomplete."
        )
    if c["refs_made_with_this_renderer_d"] is False:
        lines.append(
            "note: refs/ was made with another renderer D (refs.json made_with.renderer_d): "
            "its numbers compare this renderer D with that one"
        )
    return lines
