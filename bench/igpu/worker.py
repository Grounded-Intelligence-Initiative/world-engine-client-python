"""One run of the bench in a fresh process: one rendering path, both scenes.

    python worker.py SPEC_JSON OUT_JSON

SPEC_JSON names the path (``d``: renderer D through WGSLRenderer; ``chrome``:
the browser path through LocalRenderer, the capture page from this
directory), its options and the time limits. The run times r2_400k (the
trace's warm-up states, then the timed ones), then renders the reference
states of r2_400k and room200k and compares them with the RTX 4090
references. OUT_JSON gets one record: the adapter, the speed, the quality
numbers, the warnings, the log of world_engine_client, and the error with its
traceback when the run fails. run.py starts these and gathers the records.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare  # noqa: E402
import scene  # noqa: E402

REFS = HERE / "refs"
# The images compared: the static cameras once, the moving wrist camera at
# three states (the timed states' indices).
REF_IMAGES = [(0, "third_person"), (0, "top_down"), (0, "wrist"), (100, "wrist"), (199, "wrist")]
REF_SETS = {"vs_d_4090": "d_rgba16float_rtx4090", "vs_spark_4090": "spark_rtx4090"}
SPEED_SCENE = "r2_400k"
SCENES = ("r2_400k", "room200k")


class Log(logging.Handler):
    """world_engine_client's log lines, for the record."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record):
        if record.name.startswith("world_engine_client.render.ws"):
            return  # the capture page's socket opening and closing
        self.lines.append(f"{record.name}: {record.getMessage()}"[:2000])


def refs(name: str) -> dict:
    """{ref set key: {image name: array}} of scene ``name``."""
    out = {}
    for key, folder in REF_SETS.items():
        d = REFS / name / folder
        out[key] = {p.stem: compare.load_png(p) for p in sorted(d.glob("s*.png"))}
    return out


def open_renderer(spec: dict, name: str, cache_dir: str):
    """The renderer of the run's path for scene ``name``, and the server stand-in."""
    server = scene.Server(name, lossless=bool(spec.get("lossless")))
    scene_init, states, warmup = scene.trace(name)
    if spec["path"] == "d":
        from world_engine_client.render_wgsl import WGSLRenderer

        renderer = WGSLRenderer(
            server,
            scene_init,
            server.digests,
            cache_dir=cache_dir,
            scene_id=scene_init["scene_id"],
            adapter=spec.get("adapter") or "auto",
            color_format=spec.get("color_format"),
            allow_software=spec.get("allow_software", True),
        )
    else:
        from world_engine_client.render import LocalRenderer

        renderer = LocalRenderer(
            server,
            scene_init,
            server.digests,
            cache_dir=cache_dir,
            scene_id=scene_init["scene_id"],
            origin="igpu-bench-png" if spec.get("lossless") else "igpu-bench",
            chrome_args=spec.get("chrome_args"),
            allow_software=spec.get("allow_software", True),
            timeout=float(spec.get("timeout", 600)),
        )
    return renderer, states, warmup


def adapter_of(renderer, spec: dict) -> dict:
    info = dict(getattr(renderer, "info", {}) or {})
    if spec["path"] == "d":
        choice = renderer.choice.as_dict()
        info.update(
            vendor=choice.get("vendor"),
            description=choice.get("description"),
            search_seconds=choice.get("seconds"),
            cached=choice.get("cached"),
            trail=choice.get("trail"),
        )
    else:
        info["chrome_args"] = renderer.chrome_args
        info["lighting"] = renderer.lighting
    return info


def summary(values: list[float]) -> dict:
    s = sorted(values)
    return {
        "median": round(statistics.median(s), 3),
        "p10": round(s[len(s) // 10], 3),
        "p90": round(s[min(len(s) - 1, 9 * len(s) // 10)], 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


def reference_images(renderer, states, warmup) -> dict:
    out = {}
    for k in sorted({k for k, _ in REF_IMAGES}):
        images = renderer.render(states[warmup + k])
        for kk, cam in REF_IMAGES:
            if kk == k:
                out[f"s{k:03d}_{cam}"] = images[cam]
    return out


def timed_run(spec: dict, cache_dir: str, record: dict, images_dir: Path | None) -> None:
    """The speed scene: start, warm up, time, then its reference images."""
    t0 = time.perf_counter()
    renderer, states, warmup = open_renderer(spec, SPEED_SCENE, cache_dir)
    try:
        record["startup_s"] = round(time.perf_counter() - t0, 3)
        record["adapter"] = adapter_of(renderer, spec)
        t1 = time.perf_counter()
        renderer.render(states[0])
        record["first_frame_s"] = round(time.perf_counter() - t1, 3)
        warm_until = time.perf_counter() + float(spec["warmup_seconds"])
        done = 1
        for st in states[1:warmup]:
            if done >= 3 and time.perf_counter() > warm_until:
                break
            renderer.render(st)
            done += 1
        record["warmup_frames"] = done
        seconds, breakdown = [], {}
        limit = time.perf_counter() + float(spec["max_seconds"])
        wall = time.perf_counter()
        for st in states[warmup : warmup + int(spec["steps"])]:
            if len(seconds) >= int(spec["min_steps"]) and time.perf_counter() > limit:
                break
            a = time.perf_counter()
            renderer.render(st)
            seconds.append(time.perf_counter() - a)
            for key, value in (getattr(renderer, "last_timings", None) or {}).items():
                if isinstance(value, (int, float)):
                    breakdown.setdefault(key, []).append(float(value))
        wall = time.perf_counter() - wall
        record["timed"] = {
            "frames": len(seconds),
            "seconds": round(wall, 3),
            "steps_per_s": round(len(seconds) / wall, 3),
            "frame_ms": summary([1e3 * s for s in seconds]),
            "breakdown_ms_median": {
                k: round(statistics.median(v), 3) for k, v in breakdown.items() if v
            },
        }
        record["adapter"] = adapter_of(renderer, spec)  # with the measured rate of D
        quality(renderer, states, warmup, SPEED_SCENE, record, images_dir)
    finally:
        renderer.close()


def quality(renderer, states, warmup, name: str, record: dict, images_dir: Path | None) -> None:
    images = reference_images(renderer, states, warmup)
    entry = {"images": {k: compare.digest(v) for k, v in images.items()}}
    for key, ref in refs(name).items():
        if ref:
            entry[key] = compare.compare(ref, images)
    record.setdefault("quality", {})[name] = entry
    if images_dir is not None:
        for k, v in images.items():
            compare.save_png(images_dir / name / f"{k}.png", v)


def quality_run(spec: dict, cache_dir: str, record: dict, images_dir: Path | None) -> None:
    for name in SCENES:
        if name == SPEED_SCENE and spec.get("timed", True):
            continue  # its images came after the timed frames
        t0 = time.perf_counter()
        renderer, states, warmup = open_renderer(spec, name, cache_dir)
        try:
            record.setdefault("scene_startup_s", {})[name] = round(time.perf_counter() - t0, 3)
            if "adapter" not in record:
                record["adapter"] = adapter_of(renderer, spec)
            quality(renderer, states, warmup, name, record, images_dir)
        finally:
            renderer.close()


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    images_dir = Path(spec["images_dir"]) if spec.get("images_dir") else None
    record: dict = {"path": spec["path"], "label": spec["label"], "ok": False}
    log = Log()
    logging.getLogger("world_engine_client").addHandler(log)
    logging.getLogger("world_engine_client").setLevel(logging.INFO)
    started = time.perf_counter()
    if hasattr(os, "getloadavg"):  # other work on this machine slows a run down
        record["load_average_at_start"] = [round(x, 2) for x in os.getloadavg()]
    cache_dir = spec.get("cache_dir") or tempfile.mkdtemp(prefix="igpu-bench-cache-")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if spec.get("timed", True):
                timed_run(spec, cache_dir, record, images_dir)
            quality_run(spec, cache_dir, record, images_dir)
            record["ok"] = True
        except BaseException as exc:  # noqa: BLE001 - every failure goes in the record
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "code": getattr(exc, "code", None),
                "reason": getattr(exc, "reason", None),
                "fixes": getattr(exc, "fixes", None),
                "traceback": traceback.format_exc(),
            }
    record["warnings"] = [
        {"category": w.category.__name__, "message": str(w.message)} for w in caught
    ]
    record["log"] = log.lines
    record["seconds"] = round(time.perf_counter() - started, 3)
    if hasattr(os, "getloadavg"):
        record["load_average_at_end"] = [round(x, 2) for x in os.getloadavg()]
    out.write_text(json.dumps(record, indent=1, default=str))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
