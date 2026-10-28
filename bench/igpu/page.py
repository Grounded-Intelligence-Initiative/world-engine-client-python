"""The engine's capture page for the browser path's runs, from outside this
repository.

The browser path runs the capture page that capture_page/MANIFEST.json names:
a production build of the World Engine frontend (index.html and the two
scripts it loads) at one engine commit. The engine's repository is private,
so its build is not in this one. It comes as a directory, or as the zip that
pack_capture_page.py makes (``igpu-capture-page-<commit>.zip``), and every
file is checked against the manifest before a run. Renderer D's runs do not
use it.

Where the bench looks, in order: ``--capture-page DIR|ZIP``, the environment
variable IGPU_BENCH_CAPTURE_PAGE, then capture_page/ next to this file (when
someone put the build there).
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "capture_page"
MANIFEST = DEFAULT / "MANIFEST.json"
ENV = "IGPU_BENCH_CAPTURE_PAGE"


def manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def archive_name(m: dict) -> str:
    """The zip's name: the engine commit in it, so two builds do not mix."""
    return f"igpu-capture-page-{m['engine_commit'][:7]}.zip"


def locate(source: str | os.PathLike | None = None) -> Path | None:
    """The capture page's directory or zip: ``source``, else the environment
    variable, else capture_page/ when the build is there; None otherwise."""
    if source:
        return Path(source).expanduser()
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).expanduser()
    if (DEFAULT / "index.html").is_file():
        return DEFAULT
    return None


def missing(m: dict | None = None) -> str:
    """What to do when there is no capture page."""
    m = m or manifest()
    return (
        "the browser path's runs need the engine's capture page (engine commit "
        f"{m['engine_commit'][:7]}, see capture_page/MANIFEST.json), which is not in this "
        f"repository: pass --capture-page with its directory or {archive_name(m)}, or set "
        f"{ENV}. pack_capture_page.py makes the zip from the engine's frontend/dist. "
        "--paths d runs renderer D alone, which needs no page"
    )


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _entry(names: list[str], name: str) -> str | None:
    """The zip entry of manifest file ``name``: at the top, or under one
    folder (a zip of a capture_page/ directory)."""
    if name in names:
        return name
    depth = name.count("/") + 1
    found = [n for n in names if n.endswith("/" + name) and n.count("/") == depth]
    return found[0] if len(found) == 1 else None


def load(source: str | os.PathLike, dest: str | os.PathLike, m: dict | None = None):
    """The checked capture page as a directory, and what the result records of
    it. A directory is used where it is; from a zip, the manifest's files are
    written into ``dest`` under the manifest's names (nothing else in the zip
    is read). Each file must have the manifest's SHA-256."""
    m = m or manifest()
    src = Path(source)
    files = m["files"]
    if src.is_dir():
        bad = sorted(name for name, want in files.items() if _sha256(src / name) != want)
        kind, out = "directory", src
    elif src.is_file() and zipfile.is_zipfile(src):
        kind, out, bad = "zip", Path(dest), []
        with zipfile.ZipFile(src) as z:
            names = z.namelist()
            for name, want in sorted(files.items()):
                entry = _entry(names, name)
                data = z.read(entry) if entry is not None else None
                if data is None or hashlib.sha256(data).hexdigest() != want:
                    bad.append(name)
                    continue
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    else:
        raise SystemExit(f"{src} is neither a directory nor a zip of the capture page")
    if bad:
        raise SystemExit(
            f"{src} does not hold the capture page of capture_page/MANIFEST.json (engine "
            f"commit {m['engine_commit'][:7]}): {bad} missing or different"
        )
    record = {k: v for k, v in m.items() if k != "files"}
    record.update(source=str(src), kind=kind, files=files)
    return out, record
