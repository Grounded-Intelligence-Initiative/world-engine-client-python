"""Action spaces against a real engine, in scenes with several robots (design
doc 12 M3).

With several robots an action is every robot's, one after another, and each
control mode's schema names every robot's gripper dim (``gripper_indices``).
The server keeps each of them within ±1 in every mode, and so does the
client's action space. world-engine-client 0.4.2 to 0.6.0 bounded only the
last dim, so in ``osc_pose_abs`` a sampled action could fail the server's
check and Gymnasium's ``check_env`` failed. A scene with one robot keeps the
spaces it had.

Needs a World Engine checkout with scenes of several robots,
``WE_TEST_ENGINE`` (its tests/mobile_scene.py builds them), and a Python that
runs it, ``WE_TEST_ENGINE_PYTHON`` (default: this one; it needs mujoco, the
Menagerie and the we_ik extension). Each scene is served by
tests/engine_server.py in a process of its own, and this process is the
client. Without ``WE_TEST_ENGINE`` the tests are skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest

import world_engine_client as we

ENGINE = os.environ.get("WE_TEST_ENGINE")
pytestmark = pytest.mark.skipif(not ENGINE, reason="no engine checkout (WE_TEST_ENGINE)")

HERE = Path(__file__).resolve().parent
BASE_MODE = "osc_pose_delta_base_vel"
FIXED_MODES = ["osc_pose_delta", "osc_pose_abs", "ee_pos_delta"]


def robot(rid: str, preset: str, y: float = 0.0) -> dict:
    return {
        "id": rid,
        "preset": preset,
        "base_position": [0.0, y, 0.0],
        "base_orientation_quat": [1.0, 0.0, 0.0, 0.0],
    }


# The scenes of the engine's tests/test_multi_robot_serving.py and
# tests/test_client_contract.py.
SCENES = {
    "two_pandas": [robot("left", "franka_panda"), robot("right", "franka_panda", 0.9)],
    "tidybot_panda": [robot("tb", "stanford_tidybot"), robot("arm", "franka_panda", 1.4)],
    "panda": [robot("arm", "franka_panda")],
    "tidybot": [robot("tb", "stanford_tidybot")],
}


def _first_line(proc: subprocess.Popen, timeout: float) -> str:
    got: list[str] = []
    reader = threading.Thread(target=lambda: got.append(proc.stdout.readline()), daemon=True)
    reader.start()
    reader.join(timeout)
    return got[0] if got else ""


def _start(name: str, work: Path):
    engine = Path(ENGINE).resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = os.pathsep.join([str(engine), str(engine / "tests")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # nothing written into the engine checkout
    log = work / "server.log"
    with open(log, "w") as err:
        proc = subprocess.Popen(
            [
                os.environ.get("WE_TEST_ENGINE_PYTHON") or sys.executable,
                str(HERE / "engine_server.py"),
                str(work),
                json.dumps(SCENES[name]),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            cwd=str(work),
            env=env,
            text=True,
        )
    line = _first_line(proc, timeout=600)
    if not line:
        proc.kill()
        proc.wait()
        pytest.fail(f"the engine did not serve {name}:\n{log.read_text()[-3000:]}")
    reply = json.loads(line)
    assert Path(reply["engine"]) == engine / "world_engine", reply  # WE_TEST_ENGINE's engine
    return proc, reply["url"], reply["token"]


@pytest.fixture(scope="module")
def serve(tmp_path_factory):
    """scene name -> (url, token) of a server of that scene, started on first use."""
    pytest.importorskip("gymnasium")
    running: dict = {}

    def get(name: str) -> tuple[str, str]:
        if name not in running:
            running[name] = _start(name, tmp_path_factory.mktemp(name))
        return running[name][1:]

    yield get
    for proc, *_ in running.values():
        proc.stdin.close()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def _connect(serve, scene: str, mode: str | None):
    url, token = serve(scene)
    return we.connect(token, url=url, takeover=True, **({"control_mode": mode} if mode else {}))


CASES = [
    # scene, control_mode asked for (None: the default), the mode, action_dim, gripper dims
    pytest.param("two_pandas", None, "osc_pose_delta", 14, [6, 13], id="two_pandas-default"),
    pytest.param("two_pandas", "osc_pose_abs", "osc_pose_abs", 14, [6, 13], id="two_pandas-abs"),
    pytest.param("tidybot_panda", None, BASE_MODE, 17, [6, 16], id="tidybot_panda-default"),
    pytest.param(
        "tidybot_panda", "osc_pose_abs", "osc_pose_abs", 14, [6, 13], id="tidybot_panda-abs"
    ),
]


@pytest.mark.parametrize("scene,ask,mode,dim,grippers", CASES)
def test_every_gripper_dim_is_bounded_and_check_env_passes(serve, scene, ask, mode, dim, grippers):
    from gymnasium.utils.env_checker import check_env

    with _connect(serve, scene, ask) as env:
        assert env.metadata.get("robots") == [r["id"] for r in SCENES[scene]], (
            "this engine does not serve scenes with several robots (design doc 12 M3)"
        )
        assert env.control_mode == mode and env.action_dim == dim
        assert env.metadata["control_mode_schemas"][mode]["gripper_indices"] == grippers
        space = env.action_space
        rest = np.inf if mode == "osc_pose_abs" else 1.0
        high = [1.0 if i in grippers else rest for i in range(dim)]
        assert space.shape == (dim,) and space.dtype == np.float32
        assert space.high.tolist() == high and space.low.tolist() == [-v for v in high]
        env.reset(seed=0)
        space.seed(0)
        for _ in range(10):
            env.step(space.sample())  # the server takes each
        with warnings.catch_warnings():
            # The checker's advice on unbounded and non-normalized boxes.
            warnings.simplefilter("ignore", UserWarning)
            check_env(env.as_gymnasium(), skip_render_check=True)


@pytest.mark.parametrize("scene", ["panda", "tidybot"])
def test_one_robot_keeps_the_action_spaces_of_0_6_0(serve, scene):
    """In every mode, the box that 0.6.0 built: the mode's bounds, the last
    dim within ±1."""
    with _connect(serve, scene, None) as env:
        modes = env.metadata["supported_control_modes"]
    assert modes == ([BASE_MODE] if scene == "tidybot" else []) + FIXED_MODES
    for mode in modes:
        with _connect(serve, scene, mode) as env:
            schema = env.metadata["control_mode_schemas"][mode]
            assert "gripper_indices" not in schema, schema
            lo, hi = schema["bounds"] or (-np.inf, np.inf)
            low = np.full(env.action_dim, lo, np.float32)
            high = np.full(env.action_dim, hi, np.float32)
            low[-1], high[-1] = -1, 1
            space = env.action_space
            assert np.array_equal(space.low, low) and np.array_equal(space.high, high), mode
