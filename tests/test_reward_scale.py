"""connect(normalize_reward=...): a staged reward in points, other rewards as sent.

The server divides a staged reward by the task's total points and sends the
total with every step, in info["task_progress"]["reward"]["max"]. The episodes
below follow the engine's rules (design doc 13, every stage scores): the points
of a stage skipped at reset come with the first step, the failing step takes the
penalty, and an episode decided at reset is paid by its one frozen step. The
real engine round trip is tested in the server repository
(tests/test_remote_reward_scale.py).
"""

import math

import numpy as np
import pytest
from conftest import ENV, OBS

import world_engine_client as wec


def totals(earned, *, penalty=0.0, total=4.0):
    return {"version": 1, "earned": earned, "max": total, "partial": 0.0, "penalty": penalty}


def progress(earned, *, outcome="running", **kw):
    return {"outcome": outcome, "reward": totals(earned, **kw)}


# A staged episode out of 4 points with a penalty of 0.75: 0.5 skipped at reset
# and paid on step 1, partial credit on step 3, a stage on step 4, and the
# failure on step 5. As (reward, task_progress, terminated).
FAILS = [
    (0.125, progress(0.5), False),
    (0.0, progress(0.5), False),
    (0.09375, progress(0.875), False),
    (0.28125, progress(2.0), False),
    (-0.1875, progress(2.0, outcome="failed", penalty=0.75), True),
]
# Both stages (2 and 1 points) hold at reset: the episode is decided then, and
# its one step pays the 3 points with the reset's info, without the penalty.
AT_RESET = [(1.0, progress(3.0, outcome="skipped_initial", total=3.0), True)]


def serve(stub, steps, *, reward_type="staged", reset_progress=None):
    """Connect and reset report ``reward_type``; each step answers with the next
    of ``steps``, (reward, task_progress or None, terminated)."""
    env = {**ENV, "reward_type": reward_type}
    if reward_type is None:
        del env["reward_type"]
    queue = list(steps)

    def connect(body):
        return {"ok": True, "session_id": "sess", "proto": 3, "instance_id": "inst", "env": env}

    def reset(body):
        queue[:] = steps
        info = {"episode_id": "e1", "step_id": 0}
        if reset_progress is not None:
            info["task_progress"] = reset_progress
        return {"ok": True, "obs": OBS, "info": info, "env": env}

    def step(body):
        reward, task_progress, terminated = queue.pop(0)
        info = {"episode_id": "e1", "step_id": len(steps) - len(queue)}
        if task_progress is not None:
            info["task_progress"] = task_progress
        return {
            "ok": True,
            "obs": OBS,
            "info": info,
            "reward": reward,
            "terminated": terminated,
            "truncated": False,
        }

    stub.responders.update(
        {"/control/connect": connect, "/control/reset": reset, "/control/step": step}
    )


def episode(env, n):
    """reset, then n steps: (rewards, last info)."""
    _, info = env.reset(seed=0)
    rewards = []
    for _ in range(n):
        _, reward, terminated, truncated, info = env.step(np.zeros(4))
        rewards.append(reward)
        if terminated or truncated:
            break
    return rewards, info


def test_a_staged_reward_comes_back_as_sent_by_default(stub):
    serve(stub, FAILS)
    with wec.connect("secret", url=stub.url) as env:
        assert env.normalize_reward is True
        rewards, info = episode(env, 5)
    assert rewards == [r for r, _, _ in FAILS]
    reward = info["task_progress"]["reward"]
    assert math.isclose(sum(rewards), (reward["earned"] - reward["penalty"]) / reward["max"])


@pytest.mark.parametrize(
    "steps,reset_progress",
    [(FAILS, progress(0.5)), (AT_RESET, AT_RESET[0][1])],
    ids=["failed", "decided-at-reset"],
)
def test_points_add_up_to_earned_less_the_penalty(stub, steps, reset_progress):
    serve(stub, steps, reset_progress=reset_progress)
    with wec.connect("secret", url=stub.url, normalize_reward=False) as env:
        assert env.normalize_reward is False
        rewards, info = episode(env, 10)
    reward = info["task_progress"]["reward"]
    # Each step pays what it earned, in points: the reward times the total.
    assert rewards == [r * p["reward"]["max"] for r, p, _ in steps]
    assert all(type(r) is float for r in rewards)
    assert math.isclose(sum(rewards), reward["earned"] - reward["penalty"])
    if steps is FAILS:
        assert rewards == [0.5, 0.0, 0.375, 1.125, -0.75]
    else:
        assert rewards == [3.0] and reward["penalty"] == 0.0


@pytest.mark.parametrize(
    "reward_type,task_progress",
    [
        ("sparse", None),
        ("dense", None),
        # A staged task with a sparse or dense reward reports its stages, but
        # no points.
        ("sparse", {"outcome": "success", "current_stage": None}),
        ("dense", {"outcome": "running", "current_stage": "reach"}),
        # A server from before staged rewards sends no reward_type.
        (None, None),
    ],
    ids=["sparse", "dense", "sparse-staged-task", "dense-staged-task", "older-server"],
)
def test_sparse_and_dense_rewards_are_never_scaled(stub, reward_type, task_progress):
    steps = [(2.5, task_progress, False), (0.375, task_progress, True)]
    for normalize in (True, False):
        serve(stub, steps, reward_type=reward_type)
        with wec.connect("secret", url=stub.url, normalize_reward=normalize) as env:
            rewards, _ = episode(env, 2)
        assert rewards == [2.5, 0.375], normalize


@pytest.mark.parametrize(
    "reward_type,points",
    [("dense", False), ("sparse", False), (None, True)],
    ids=["dense", "sparse", "older-server"],
)
def test_the_env_reward_type_decides_over_the_info(stub, reward_type, points):
    """Only a staged reward is in points, even if a step's info had totals.
    A server without reward_type sends totals only for a staged reward."""
    steps = [(0.125, progress(0.5), False), (0.375, progress(2.0), True)]
    serve(stub, steps, reward_type=reward_type)
    with wec.connect("secret", url=stub.url, normalize_reward=False) as env:
        rewards, _ = episode(env, 2)
    assert rewards == ([0.5, 1.5] if points else [0.125, 0.375])


@pytest.mark.parametrize(
    "task_progress",
    [
        None,
        {"outcome": "running"},
        {"outcome": "running", "reward": None},
        {"outcome": "running", "reward": [1.0]},
        progress(0.5, total="4"),
        progress(0.5, total=True),
        progress(0.5, total=0),
        progress(0.5, total=-4.0),
        progress(0.5, total=float("nan")),
        progress(0.5, total=1e308),
    ],
    ids=["no-progress", "no-totals", "null-totals", "list-totals", "string", "bool", "zero",
         "negative", "nan", "overflow"],
)  # fmt: skip
def test_a_staged_reward_without_a_usable_total_needs_a_reset(stub, task_progress):
    serve(stub, [(10.0, task_progress, False), (0.0, progress(0.5), False)])
    with wec.connect("secret", url=stub.url, normalize_reward=False) as env:
        env.reset()
        with pytest.raises(wec.ProtocolError, match="normalize_reward=True") as caught:
            env.step(np.zeros(4))
        assert "task_progress" in str(caught.value)
        sent = stub.paths().count("/control/step")
        with pytest.raises(wec.RecoveryRequiredError):
            env.step(np.zeros(4))
        assert stub.paths().count("/control/step") == sent
        env.normalize_reward = True
        env.reset()
        assert env.step(np.zeros(4))[1] == 10.0, "rewards as sent need no total"


@pytest.mark.parametrize("value", ["false", None, 0, 1, np.False_])
def test_normalize_reward_takes_a_bool_only(stub, value):
    with pytest.raises(TypeError, match="normalize_reward must be True or False"):
        wec.connect("secret", url=stub.url, normalize_reward=value)
    assert stub.calls == [], "rejected before any request"


def test_gymnasium_and_its_wrappers_see_the_points(stub):
    gym = pytest.importorskip("gymnasium")
    serve(stub, FAILS)
    remote = wec.connect("secret", url=stub.url, normalize_reward=False)
    env = gym.wrappers.RecordEpisodeStatistics(remote.as_gymnasium())
    env.reset(seed=0)
    for _ in range(5):
        *_, terminated, truncated, info = env.step(np.zeros(4, np.float32))
    assert terminated
    reward = info["task_progress"]["reward"]
    assert info["episode"]["r"] == pytest.approx(reward["earned"] - reward["penalty"])
    assert info["episode"]["r"] == pytest.approx(1.25)
    env.close()
