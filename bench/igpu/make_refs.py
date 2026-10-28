"""Make the bench's reference images; they were made on an RTX 4090 and are
committed in refs/. Run again only to replace them, on an RTX 4090:

    python bench/igpu/make_refs.py --capture-page DIR|ZIP [--sets d] [--force]

For each scene: renderer D with its default colour target (rgba16float on a
GPU), and Spark through the capture page (page.py: the engine's build, from
outside this repository) with its JPEG encoding replaced by PNG (the same
pixels as a readback). Each is rendered twice, in two renderers, and must come
out the same both times. refs/refs.json records what made them, and
``made_with`` the client: its version, the checkout's git description and
the digest of renderer D's files (client.py), which a test checks against
this checkout. After renderer D changes, ``--sets d`` makes D's images again
and keeps Spark's.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import client  # noqa: E402
import compare  # noqa: E402
import machine  # noqa: E402
import page  # noqa: E402
import worker  # noqa: E402


def render(spec: dict, name: str, cache_dir: str) -> tuple[dict, dict]:
    renderer, states, warmup = worker.open_renderer(spec, name, cache_dir)
    try:
        return worker.reference_images(renderer, states, warmup), worker.adapter_of(renderer, spec)
    finally:
        renderer.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="on another GPU than an RTX 4090")
    ap.add_argument("--capture-page", default=None, metavar="DIR|ZIP", help="as run.py's (page.py)")
    ap.add_argument(
        "--sets",
        choices=("all", "d"),
        default="all",
        help="d: renderer D's images only, Spark's kept (after renderer D changed)",
    )
    a = ap.parse_args()
    where = page.locate(a.capture_page) if a.sets == "all" else None
    if where is None and a.sets == "all":
        ap.error(page.missing())
    os.environ["WE_RENDER_BROWSER"] = "shell"  # the pinned headless shell, not a local Chrome
    found = client.info()
    if found["matches_checkout"] is False:
        ap.error(f"the installed world-engine-client is not this checkout's: {found['differs']}")
    work = Path(tempfile.mkdtemp(prefix="igpu-refs-"))
    try:
        return make(a, where, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make(a, where: Path | None, work: Path) -> int:
    cache_dir = str(work / "render-cache")
    sets = {"d_rgba16float_rtx4090": {"path": "d", "label": "renderer D", "adapter": "auto"}}
    kept = json.loads((worker.REFS / "refs.json").read_text())
    if where is not None:
        page_dir, _ = page.load(where, work / "capture_page")
        sets["spark_rtx4090"] = {
            "path": "chrome",
            "label": "Spark, PNG",
            "lossless": True,
            "capture_page": str(page_dir),
        }
    found_client = client.info()
    record: dict = {
        "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "images": worker.REF_IMAGES,
        "made_with": {
            "world-engine-client": found_client["installed"]["version"],
            "git": (found_client["checkout"] or {}).get("git"),
            "renderer_d": found_client["renderer_d"],
            "matches_checkout": found_client["matches_checkout"],
        },
    }
    if where is None:
        record["made_with"]["spark_kept_from"] = kept.get("made")
        record["sets"] = {
            name: {"spark_rtx4090": kept["sets"][name]["spark_rtx4090"]} for name in worker.SCENES
        }
    found = machine.collect()  # what made them, without paths or the host's name
    record["machine"] = {
        "os": {k: found["os"].get(k) for k in ("system", "release", "distribution", "machine")},
        "cpu": {k: found["cpu"].get(k) for k in ("model", "logical_cores")},
        "wgpu_adapters": found["gpus"].get("wgpu_adapters"),
        "nvidia_smi": (found["gpus"].get("nvidia_smi") or {}).get("stdout"),
        "packages": {k: v for k, v in found["packages"].items() if not k.endswith("_file")},
        "python": found["python"]["version"].split()[0],
    }
    for name in worker.SCENES:
        for folder, spec in sets.items():
            first, adapter = render(spec, name, cache_dir)
            again, _ = render(spec, name, cache_dir)
            same = {k: compare.digest(first[k]) == compare.digest(again[k]) for k in first}
            if not all(same.values()):
                raise SystemExit(f"{name} {folder}: two renders differ: {same}")
            who = adapter.get("adapter") or ""
            if "4090" not in who and not a.force:
                raise SystemExit(f"{folder} rendered on {who}, not an RTX 4090 (--force)")
            if spec["path"] == "d" and adapter.get("color_format") != "rgba16float":
                raise SystemExit(f"renderer D used {adapter.get('color_format')}, not rgba16float")
            out = worker.REFS / name / folder
            shutil.rmtree(out, ignore_errors=True)
            for k, image in first.items():
                compare.save_png(out / f"{k}.png", image)
            entry = {"adapter": adapter, "images": {k: compare.digest(v) for k, v in first.items()}}
            record.setdefault("sets", {}).setdefault(name, {})[folder] = entry
            print(f"{name} {folder}: {who}, {len(first)} images, two renders the same", flush=True)
        d = worker.REFS / name
        diff = compare.compare(
            {p.stem: compare.load_png(p) for p in (d / "spark_rtx4090").glob("*.png")},
            {p.stem: compare.load_png(p) for p in (d / "d_rgba16float_rtx4090").glob("*.png")},
        )
        record["sets"][name]["d_vs_spark_on_the_4090"] = diff
        print(f"{name}: D against Spark on the RTX 4090: {diff['mean']}/255", flush=True)
    (worker.REFS / "refs.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
