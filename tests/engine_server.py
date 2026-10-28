"""A World Engine server for tests/test_engine_spaces.py: a scene with the
given robots, built the way the engine's own tests build one
(tests/mobile_scene.py there), in a MuJoCo session served over HTTP and paused
until a controller steps it, as a hosted gym is.

    python engine_server.py WORK_DIR ROBOTS_JSON

ROBOTS_JSON lists the scene's robots as world_composition.json does. Run it
with the engine's Python and with the engine checkout and its tests/ on
PYTHONPATH. Once the server listens it prints one line on stdout,
``{"url": ..., "token": ..., "engine": <the world_engine package it runs>}``,
and serves until its stdin closes. Everything else goes to stderr.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path


def main() -> int:
    work, robots = Path(sys.argv[1]), json.loads(sys.argv[2])
    reply_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # the engine's output, C's too
    import world_engine
    from mobile_scene import build
    from world_engine.serving.remote_control import RemoteControlManager
    from world_engine.serving.sim_session import SimSession
    from world_engine.serving.viewer_server import Capabilities, ViewerServer

    build(work, robots, scene_id="spaces")
    session = SimSession("spaces", None, scenes_dir=str(work / "scenes_spaces"), physics="mujoco")
    session.core.reset(seed=0)
    token = secrets.token_urlsafe(24)
    server = ViewerServer(
        capabilities=Capabilities(mode="simulator", control_token=token),
        host="127.0.0.1",
        port=0,
        dist_dir=work,
    )
    session.attach(server)
    RemoteControlManager(session).register(server)
    session.keep_paused = True
    session.set_remote_mode(True)
    port = server.start()
    engine = str(Path(world_engine.__file__).resolve().parent)
    with os.fdopen(reply_fd, "w") as out:
        out.write(json.dumps({"url": f"http://127.0.0.1:{port}", "token": token, "engine": engine}))
        out.write("\n")
    try:
        sys.stdin.read()
    finally:
        server.stop()
        session.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
