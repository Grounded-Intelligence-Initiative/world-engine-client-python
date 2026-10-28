"""The integrated-GPU bench: renderer D against the browser path, on this machine.

    python bench/igpu/run.py [--adapter name:Intel] [--rounds 3] [--out FILE]

Each path renders the same splat-only scenes from the same cameras and trace:
r2_400k, timed, and room200k, for quality (scene.py says why two). A round
runs every path once, each in a fresh process (worker.py), and the order
turns from round to round. After the rounds, one more browser-path run reads
its images back losslessly (PNG in place of JPEG) to compare Spark here with
Spark on the RTX 4090. Everything goes into one JSON file: the machine, the
versions, each run's adapter, speed, quality numbers, warnings and errors,
and a summary against the thresholds of README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import machine  # noqa: E402
import page  # noqa: E402

FORMAT = 1
PATHS = {
    "d": {"path": "d", "label": "renderer D, default colour target"},
    "chrome": {"path": "chrome", "label": "browser path: headless Chrome, JPEG"},
    "d_rgba8": {"path": "d", "color_format": "rgba8unorm", "label": "renderer D, rgba8unorm"},
}
# Design doc 16 §9.4 and §9.6 (M-D1), task book §8.1 item 4 (b): on room200k.
THRESHOLDS = {"d_vs_d_4090": 1.5, "d_vs_spark_4090": 3.0}
SOFTWARE_WORDS = ("llvmpipe", "lavapipe", "swiftshader", "softpipe", "basic render", "warp")
VENDOR_WORDS = (
    ("intel", ("intel",)),
    ("nvidia", ("nvidia", "geforce", "quadro", "rtx")),
    ("amd", ("amd", "radeon")),
    ("apple", ("apple",)),
    ("qualcomm", ("qualcomm", "adreno")),
    ("arm", ("mali",)),
)


def gpu_class(text: str | None) -> str | None:
    """What kind of adapter a name says: software, or the GPU's vendor."""
    t = (text or "").lower()
    if any(w in t for w in SOFTWARE_WORDS) or "software" in t:
        return "software"
    for vendor, words in VENDOR_WORDS:
        if any(w in t for w in words):
            return vendor
    return None


def round_order(keys: list[str], rnd: int) -> list[str]:
    """The paths of round ``rnd`` (from 1): turned by one each round, so each
    path runs early in one round and late in another."""
    shift = (rnd - 1) % len(keys)
    return keys[shift:] + keys[:shift]


def run_worker(spec: dict, env: dict, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="igpu-bench-") as tmp:
        spec_file, out_file = Path(tmp) / "spec.json", Path(tmp) / "out.json"
        spec_file.write_text(json.dumps(spec))
        started = time.perf_counter()
        try:
            done = subprocess.run(
                [sys.executable, str(HERE / "worker.py"), str(spec_file), str(out_file)],
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
            code, stderr = done.returncode, done.stderr
        except subprocess.TimeoutExpired as exc:
            code = None
            stderr = (exc.stderr or b"").decode("utf-8", "replace") if exc.stderr else ""
            stderr += f"\n[bench] the run did not finish within {timeout:.0f} s"
        wall = round(time.perf_counter() - started, 3)
        if out_file.exists():
            record = json.loads(out_file.read_text())
        else:
            record = {"path": spec["path"], "label": spec["label"], "ok": False}
            record["error"] = {"type": "WorkerCrashed", "message": stderr[-4000:]}
    record.update(exit_code=code, wall_s=wall, stderr_tail=stderr[-4000:])
    return record


def one_line(record: dict) -> str:
    a = record.get("adapter") or {}
    who = a.get("adapter") or "?"
    if record["path"] == "d":
        who += f" ({a.get('adapter_type')}, {a.get('backend')}, step {a.get('step')})"
        who += f", {a.get('color_format')}"
    else:
        who += f" ({a.get('browser')} {a.get('browser_version')})"
    if not record.get("ok"):
        err = record.get("error") or {}
        return f"{record['key']}: FAILED on {who}: {err.get('type')}: {err.get('message')}"[:600]
    parts = [f"{record['key']}: {who}"]
    if "timed" in record:
        t = record["timed"]
        parts.append(f"{t['steps_per_s']:.2f} steps/s ({t['frames']} frames)")
    room = (record.get("quality") or {}).get("room200k") or {}
    for key in ("vs_d_4090", "vs_spark_4090"):
        if key in room:
            parts.append(f"room200k {key} {room[key]['mean']:.3f}")
    if record.get("warnings"):
        parts.append(f"{len(record['warnings'])} warning(s)")
    return ", ".join(parts)


def stats(values: list[float]) -> dict:
    return {
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "per_run": [round(v, 3) for v in values],
    }


def summarize(runs: list[dict]) -> dict:
    by_key: dict[str, list[dict]] = {}
    for r in runs:
        by_key.setdefault(r["key"], []).append(r)
    out: dict = {"paths": {}}
    for key, rs in by_key.items():
        ok = [r for r in rs if r.get("ok")]
        entry: dict = {"runs": len(rs), "ok": len(ok)}
        rates = [r["timed"]["steps_per_s"] for r in ok if "timed" in r]
        if rates:
            entry["steps_per_s"] = stats(rates)
            entry["frame_ms_median"] = stats([r["timed"]["frame_ms"]["median"] for r in ok])
        quality: dict = {}
        for name in ("r2_400k", "room200k"):
            per = [r["quality"][name] for r in ok if name in (r.get("quality") or {})]
            if not per:
                continue
            q = {}
            for ref in ("vs_d_4090", "vs_spark_4090"):
                means = [p[ref]["mean"] for p in per if ref in p and "mean" in p[ref]]
                if means:
                    q[ref] = {
                        "mean": stats(means),
                        "worst_image": max(p[ref]["worst_image"] for p in per if ref in p),
                        "max_pixel_diff": max(p[ref]["max"] for p in per if ref in p),
                        "frac_gt2": max(p[ref]["frac_gt2"] for p in per if ref in p),
                    }
            digests = {json.dumps(p["images"], sort_keys=True) for p in per}
            q["identical_across_runs"] = len(digests) == 1 if len(per) > 1 else None
            quality[name] = q
        entry["quality"] = quality
        entry["adapters"] = sorted({one_adapter(r) for r in ok})
        entry["warnings"] = sorted({w["message"] for r in rs for w in r.get("warnings", [])})
        out["paths"][key] = entry
    out["d_over_chrome"] = speed_ratio(by_key)
    out["same_gpu"] = same_gpu(by_key)
    out["checks"] = checks(out)
    out["pass"] = all(c["pass"] for c in out["checks"] if c["pass"] is not None)
    return out


def one_adapter(r: dict) -> str:
    a = r.get("adapter") or {}
    if r["path"] == "d":
        return f"{a.get('adapter')} ({a.get('adapter_type')}, {a.get('backend')})"
    return f"{a.get('adapter')} [{a.get('browser')} {a.get('browser_version')}]"


def speed_ratio(by_key: dict) -> dict | None:
    d = [r for r in by_key.get("d", []) if r.get("ok") and "timed" in r]
    c = [r for r in by_key.get("chrome", []) if r.get("ok") and "timed" in r]
    if not d or not c:
        return None
    dm = statistics.median(r["timed"]["steps_per_s"] for r in d)
    cm = statistics.median(r["timed"]["steps_per_s"] for r in c)
    rounds = {}
    for r in d + c:
        rounds.setdefault(r["round"], {})[r["path"]] = r["timed"]["steps_per_s"]
    per_round = [
        round(v["d"] / v["chrome"], 3)
        for _, v in sorted(rounds.items())
        if len(v) == 2 and v["chrome"]
    ]
    return {"median_ratio": round(dm / cm, 3) if cm else None, "per_round": per_round}


def same_gpu(by_key: dict) -> dict:
    d = {
        gpu_class((r.get("adapter") or {}).get("adapter"))
        for r in by_key.get("d", [])
        if r.get("ok")
    }
    c = {
        gpu_class((r.get("adapter") or {}).get("adapter"))
        for r in by_key.get("chrome", [])
        if r.get("ok")
    }
    known = None not in d | c and d and c
    return {
        "d": sorted(x or "?" for x in d),
        "chrome": sorted(x or "?" for x in c),
        "same": (d == c and len(d) == 1) if known else None,
    }


def checks(summary: dict) -> list[dict]:
    paths = summary["paths"]
    d = paths.get("d", {})
    room = d.get("quality", {}).get("room200k", {})
    out = []
    names = {
        "vs_d_4090": "renderer D on the RTX 4090",
        "vs_spark_4090": "Spark (lossless) on the RTX 4090",
    }
    for ref, limit in (("vs_d_4090", "d_vs_d_4090"), ("vs_spark_4090", "d_vs_spark_4090")):
        value = room.get(ref, {}).get("mean", {}).get("max")  # the worst run's mean
        out.append(
            {
                "name": f"room200k: renderer D against {names[ref]}, mean |diff| <= "
                f"{THRESHOLDS[limit]}/255",
                "value": value,
                "limit": THRESHOLDS[limit],
                "pass": None if value is None else value <= THRESHOLDS[limit],
            }
        )
    fog = d.get("quality", {}).get("r2_400k", {})
    value = fog.get("vs_spark_4090", {}).get("mean", {}).get("max")
    out.append(
        {
            "name": "r2_400k (a uniform fog: only whether D draws at all): renderer D against "
            "Spark on the RTX 4090, mean |diff| <= 3/255",
            "value": value,
            "limit": 3.0,
            "pass": None if value is None else value <= 3.0,
        }
    )
    ratio = summary.get("d_over_chrome") or {}
    same = summary.get("same_gpu", {}).get("same")
    value = ratio.get("median_ratio")
    out.append(
        {
            "name": "r2_400k: renderer D not slower than the browser path (median steps/s ratio "
            ">= 1)" + ("" if same else "; only meaningful when both used the same GPU"),
            "value": value,
            "limit": 1.0,
            "pass": None if value is None else value >= 1.0,
            "same_gpu": same,
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--paths", default="d,chrome,d_rgba8", help="of: " + ", ".join(PATHS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument(
        "--adapter",
        default=None,
        help="renderer D's adapter: auto, low-power, software or name:<part of its name> "
        "(default WE_RENDER_ADAPTER, else auto)",
    )
    ap.add_argument(
        "--browser",
        choices=("shell", "auto", "chrome"),
        default="shell",
        help="the browser path's browser (WE_RENDER_BROWSER): the pinned headless shell, the "
        "same build on every machine (default); auto: an installed Google Chrome first",
    )
    ap.add_argument(
        "--capture-page",
        default=None,
        metavar="DIR|ZIP",
        help="the engine's capture page for the browser path: its directory or "
        f"igpu-capture-page-<commit>.zip (default {page.ENV}, else capture_page/)",
    )
    ap.add_argument(
        "--chrome-args", default=None, help="Chrome's flags, one string (default: the client's)"
    )
    ap.add_argument(
        "--no-software", action="store_true", help="refuse software adapters on both paths"
    )
    ap.add_argument(
        "--d-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an environment variable for renderer D's runs only (repeatable)",
    )
    ap.add_argument(
        "--steps", type=int, default=200, help="timed states of the trace (at most 200)"
    )
    ap.add_argument("--min-steps", type=int, default=5)
    ap.add_argument(
        "--max-seconds", type=float, default=60.0, help="time limit of the timed frames"
    )
    ap.add_argument("--warmup-seconds", type=float, default=10.0)
    ap.add_argument("--run-timeout", type=float, default=1800.0, help="seconds per run")
    ap.add_argument(
        "--no-lossless", action="store_true", help="skip the PNG run of the browser path"
    )
    ap.add_argument("--no-images", action="store_true", help="keep no images of the runs")
    ap.add_argument(
        "--keep-cache",
        action="store_true",
        help="keep the run's temporary directory (render cache, capture page) at the end",
    )
    ap.add_argument("--label", default="", help="free text for the result, e.g. the machine")
    ap.add_argument(
        "--out", default=None, help="the result file (default igpu_bench_<host>_<time>.json)"
    )
    a = ap.parse_args(argv)

    try:
        from world_engine_client import render
    except ImportError as exc:
        ap.error(f"world-engine-client is not installed: {exc}")
    if not hasattr(render, "browsers_dir"):
        ap.error(
            "this world-engine-client predates the bench (no pinned headless shell): install "
            "0.7.0 or later, pip install -U 'world-engine-client[render]'"
        )
    keys = [k.strip() for k in a.paths.split(",") if k.strip()]
    unknown = [k for k in keys if k not in PATHS]
    if unknown:
        ap.error(f"unknown paths {unknown}; known: {list(PATHS)}")
    needs_page = any(PATHS[k]["path"] == "chrome" for k in keys)
    where = page.locate(a.capture_page) if needs_page else None
    if needs_page and where is None:
        ap.error(page.missing())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(a.out or f"igpu_bench_{platform.node() or 'host'}_{stamp}.json").resolve()
    images = None if a.no_images else out.with_name(out.stem + "_images")
    adapter = a.adapter or os.environ.get("WE_RENDER_ADAPTER") or "auto"
    d_env = dict(kv.split("=", 1) for kv in a.d_env)
    chrome_args = shlex.split(a.chrome_args) if a.chrome_args else None
    # The render cache (the scenes' assets, the page's files) and the page
    # taken from a zip: one temporary directory, removed at the end.
    work = Path(tempfile.mkdtemp(prefix="igpu-bench-"))
    try:
        if needs_page:
            page_dir, page_record = page.load(where, work / "capture_page")
        else:
            page_dir, page_record = None, {"needed": False}
        print(f"[bench] collecting machine information -> {out}", flush=True)
        result: dict = {
            "bench": "igpu",
            "format": FORMAT,
            "label": a.label,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": sys.argv[1:] if argv is None else argv,
            "config": {
                "paths": keys,
                "rounds": a.rounds,
                "adapter": adapter,
                "browser": a.browser,
                "chrome_args": chrome_args,
                "allow_software": not a.no_software,
                "d_env": d_env,
                "steps": a.steps,
                "min_steps": a.min_steps,
                "max_seconds": a.max_seconds,
                "warmup_seconds": a.warmup_seconds,
                "thresholds": THRESHOLDS,
            },
            "capture_page": page_record,
            "machine": machine.collect(),
            "runs": [],
        }
        for ad in result["machine"].get("gpus", {}).get("wgpu_adapters") or []:
            if isinstance(ad, dict):
                print(
                    f"[bench] wgpu adapter: {ad.get('device')} ({ad.get('adapter_type')}, "
                    f"{ad.get('backend_type')}) {ad.get('description') or ''}",
                    flush=True,
                )
        base = {
            "adapter": adapter,
            "chrome_args": chrome_args,
            "allow_software": not a.no_software,
            "steps": a.steps,
            "min_steps": a.min_steps,
            "max_seconds": a.max_seconds,
            "warmup_seconds": a.warmup_seconds,
            "cache_dir": str(work / "render-cache"),
            "capture_page": str(page_dir) if page_dir is not None else None,
        }

        def launch(key: str, spec: dict, rnd: int, slot: int) -> dict:
            env = dict(os.environ)
            if spec["path"] == "d":
                env.update(d_env)
            else:
                env["WE_RENDER_BROWSER"] = a.browser
            if images is not None:
                spec["images_dir"] = str(images / f"round{rnd}_{key}")
            record = run_worker(spec, env, a.run_timeout)
            record.update(key=key, round=rnd, slot=slot)
            print(f"[bench] round {rnd}, " + one_line(record), flush=True)
            result["runs"].append(record)
            out.write_text(json.dumps(result, indent=1, default=str))  # what there is so far
            return record

        for rnd in range(1, a.rounds + 1):
            for slot, key in enumerate(round_order(keys, rnd)):
                launch(key, {**base, **PATHS[key], "timed": True}, rnd, slot)
        if "chrome" in keys and not a.no_lossless:
            spec = {
                **base,
                **PATHS["chrome"],
                "label": "browser path, PNG readback (quality only)",
                "lossless": True,
                "timed": False,
            }
            launch("chrome_png", spec, 0, 0)

        result["summary"] = summarize(result["runs"])
        result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        out.write_text(json.dumps(result, indent=1, default=str))
        print(f"[bench] wrote {out}", flush=True)
        for c in result["summary"]["checks"]:
            verdict = {True: "pass", False: "FAIL", None: "n/a"}[c["pass"]]
            print(f"[bench] {verdict}: {c['name']}: {c['value']}", flush=True)
    finally:
        if a.keep_cache:
            print(f"[bench] kept {work}", flush=True)
        else:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
