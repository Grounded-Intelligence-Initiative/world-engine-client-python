"""Pack the engine's capture page build into the zip the bench takes.

    python bench/igpu/pack_capture_page.py DIST [--out FILE]

DIST is the engine's frontend/dist after ``npm run build`` at the commit that
capture_page/MANIFEST.json names. Every file the manifest lists must be there
with its SHA-256. The zip holds those files, MANIFEST.json and LICENSES.md,
with fixed dates, so packing the same build again gives the same bytes. The
bench then takes it with ``--capture-page FILE``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import page  # noqa: E402


def pack(dist: Path, out: Path, m: dict | None = None, extra: dict | None = None) -> str:
    """Write the zip; returns its SHA-256. ``extra`` maps names in the zip to
    files to add as they are (default: MANIFEST.json and LICENSES.md)."""
    m = m or page.manifest()
    if extra is None:
        extra = {"MANIFEST.json": page.MANIFEST, "LICENSES.md": page.DEFAULT / "LICENSES.md"}
    page.load(dist, dest=dist, m=m)  # every file there, with its SHA-256
    members = {name: Path(dist) / name for name in m["files"]}
    members.update(extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as z:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            z.writestr(info, Path(members[name]).read_bytes())
    return hashlib.sha256(out.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dist", help="the engine's frontend/dist")
    ap.add_argument("--out", default=None, help="the zip (default igpu-capture-page-<commit>.zip)")
    a = ap.parse_args(argv)
    m = page.manifest()
    out = Path(a.out or page.archive_name(m)).resolve()
    digest = pack(Path(a.dist), out, m)
    print(f"{out}\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
