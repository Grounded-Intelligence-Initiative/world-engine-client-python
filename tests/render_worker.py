"""One rendering session in a process of its own, for test_cache_processes.py
and test_render_chrome.py.

    python render_worker.py <server url> <cache dir> <result.json> platform|local
        [--chrome] [--hold <marker file>]

Chrome is the stand-in of fake_chrome.py, or with ``--chrome`` a real headless
Chrome on SwiftShader. ``--hold`` writes the marker file after the first
frame and keeps the session open until the process is killed. The result:
what the page received (sha256 per URL, stand-in only), what failed, and when
the session started and ended.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SWIFTSHADER = [
    "--headless=new",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--no-sandbox",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("cache_dir")
    parser.add_argument("out")
    parser.add_argument("mode", choices=["platform", "local"])
    parser.add_argument("--chrome", action="store_true")
    parser.add_argument("--hold")
    args = parser.parse_args()
    sim = None
    if not args.chrome:
        import fake_chrome

        sim = fake_chrome.install()
    import world_engine_client as wec

    result = {"started": time.time(), "client": wec.__file__}
    extra = {"access_provider": lambda: "scoped"} if args.mode == "platform" else {}
    options = {"cache_dir": args.cache_dir, "timeout": 120}
    if args.chrome:
        options.update(chrome_args=SWIFTSHADER, allow_software=True, auto_install=False)
    try:
        env = wec.connect("secret", url=args.url, images=True, render_options=options, **extra)
        try:
            obs, _ = env.reset(seed=0)
            result["pixel"] = [int(c) for c in obs["rgb"]["front"][0, 0]]
            if args.hold:
                Path(args.hold).write_text("ready")
                time.sleep(3600)
        finally:
            env.close()
    except Exception as exc:  # noqa: BLE001 - the test reads it
        result["error"] = f"{type(exc).__name__}: {exc}"
    if sim is not None:
        result["received"] = {u: hashlib.sha256(d).hexdigest() for u, d in sim.received.items()}
        result["failed"] = {u: str(e) for u, e in sim.failed.items()}
    result["ended"] = time.time()
    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
