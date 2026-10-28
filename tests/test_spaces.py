"""Gymnasium spaces follow the negotiated control mode, obs schema and cameras."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ENV

from world_engine_client import RemoteEnv
from world_engine_client.gym_adapter import RemoteGymEnv
from world_engine_client.spaces import build_spaces, gripper_dims


def remote(mode="ee_pos_delta", dim=4, cameras=None):
    metadata = copy.deepcopy(ENV)
    metadata["cameras"] = cameras or {}
    return SimpleNamespace(metadata=metadata, control_mode=mode, action_dim=dim)


@pytest.mark.parametrize("mode,dim", [("ee_pos_delta", 4), ("osc_pose_delta", 7)])
def test_delta_modes_are_bounded_to_unit_box(mode, dim):
    action, _ = build_spaces(remote(mode, dim).metadata, mode, dim)
    assert action.shape == (dim,) and action.dtype == np.float32
    assert (action.low == -1).all() and (action.high == 1).all()
    assert not action.contains(np.full(dim, 1.5, np.float32))


def test_absolute_pose_is_unbounded_except_gripper():
    action, _ = build_spaces(remote().metadata, "osc_pose_abs", 7)
    assert np.isinf(action.low[:6]).all() and np.isinf(action.high[:6]).all()
    assert (action.low[-1], action.high[-1]) == (-1, 1)
    assert action.contains(np.array([0.5, -0.2, 1.3, 3.0, 0, 0, 1], np.float32))
    assert not action.contains(np.array([0.5, -0.2, 1.3, 3.0, 0, 0, 1.5], np.float32))


def test_the_gripper_dims_follow_the_servers_rule():
    """gripper_indices, else gripper_index, else the last dim: the cases of the
    engine's test of the same rule (tests/test_multi_robot_serving.py)."""
    assert gripper_dims({"gripper_indices": [6, 13]}, 14) == [6, 13]
    assert gripper_dims({"gripper_index": 6}, 10) == [6]
    assert gripper_dims({}, 7) == [6]
    assert gripper_dims({"gripper_indices": [], "gripper_index": 3}, 4) == [3]


def with_schema(mode, schema):
    """ENV with ``mode``'s schema replaced by ``schema``."""
    metadata = copy.deepcopy(ENV)
    metadata["control_mode_schemas"][mode] = schema
    return metadata


def test_absolute_pose_with_two_robots_bounds_each_gripper():
    """Two robots in osc_pose_abs (the engine's scene_schema): both gripper
    dims within ±1, the targets unbounded."""
    schema = {"action_dim": 14, "bounds": None, "gripper_index": 6, "gripper_indices": [6, 13]}
    action, _ = build_spaces(with_schema("osc_pose_abs", schema), "osc_pose_abs", 14)
    for i in range(14):
        if i in (6, 13):
            assert (action.low[i], action.high[i]) == (-1, 1), i
        else:
            assert np.isinf(action.low[i]) and np.isinf(action.high[i]), i
    ok = np.array([0.5, -0.2, 1.3, 3.0, 0, 0, 1] * 2, np.float32)
    assert action.contains(ok)
    for i in (6, 13):
        bad = ok.copy()
        bad[i] = 1.5
        assert not action.contains(bad), i


def test_a_robot_on_a_mobile_base_next_to_another_is_bounded_everywhere():
    """A TidyBot next to a Panda in the TidyBot's mode: 10 values and 7, a
    gripper at 6 and at 16, every dim within ±1."""
    mode = "osc_pose_delta_base_vel"
    schema = {
        "action_dim": 17,
        "bounds": [-1.0, 1.0],
        "gripper_index": 6,
        "gripper_indices": [6, 16],
    }
    action, _ = build_spaces(with_schema(mode, schema), mode, 17)
    assert (action.low == -1).all() and (action.high == 1).all()


@pytest.mark.parametrize(
    "mode,schema",
    [
        *ENV["control_mode_schemas"].items(),
        # One TidyBot: its own mode puts the gripper at 6, base_wz last.
        ("osc_pose_delta_base_vel", {"action_dim": 10, "bounds": [-1.0, 1.0], "gripper_index": 6}),
    ],
)
def test_one_robot_keeps_the_spaces_of_0_6_0(mode, schema):
    """The box 0.6.0 built from a schema without gripper_indices: the mode's
    bounds, the last dim within ±1."""
    dim = schema["action_dim"]
    action, _ = build_spaces(with_schema(mode, schema), mode, dim)
    lo, hi = schema["bounds"] or (-np.inf, np.inf)
    low, high = np.full(dim, lo, np.float32), np.full(dim, hi, np.float32)
    low[-1], high[-1] = -1, 1
    assert np.array_equal(action.low, low) and np.array_equal(action.high, high)


def test_state_only_observation_space_and_render():
    env = RemoteGymEnv(remote())
    assert set(env.observation_space.spaces) == {"robot0_eef_pos"}
    assert env.observation_space["robot0_eef_pos"].shape == (3,)
    assert env.render_mode is None and env.metadata["render_modes"] == []
    assert env.metadata["render_fps"] == 20
    assert env.render() is None


def test_images_add_an_rgb_dict_with_per_camera_shapes():
    cameras = {"front": {"width": 320, "height": 240}, "wrist": {"width": 64, "height": 64}}
    env = RemoteGymEnv(remote(cameras=cameras))
    rgb = env.observation_space["rgb"]
    assert rgb["front"].shape == (240, 320, 3) and rgb["front"].dtype == np.uint8
    assert rgb["wrist"].shape == (64, 64, 3)
    assert env.render_mode == "rgb_array" and env.metadata["render_modes"] == ["rgb_array"]
    obs = {
        "robot0_eef_pos": np.zeros(3, np.float32),
        "rgb": {
            "front": np.zeros((240, 320, 3), np.uint8),
            "wrist": np.zeros((64, 64, 3), np.uint8),
        },
    }
    assert env.observation_space.contains(obs)


def test_adapter_moves_transport_identity_out_of_gym_info():
    obs = {"robot0_eef_pos": np.zeros(3, np.float32), "object_ids": ["cube"]}
    info = {"episode_id": "e", "instance_id": "i", "timings": {}, "success": True, "step_id": 3}
    adapted, gym_info = RemoteGymEnv._adapt(obs, info)
    assert list(adapted) == ["robot0_eef_pos"]
    assert gym_info == {"success": True, "step_id": 3, "object_ids": ["cube"]}
    assert "object_ids" in obs  # the caller's dict is not mutated


def test_remote_env_exposes_spaces_directly():
    env = RemoteEnv("http://127.0.0.1:1", "token")
    env._refresh_env(remote("osc_pose_delta", 7).metadata | {"action_dim": 7})
    env.control_mode = "osc_pose_delta"
    assert env.action_space.shape == (7,)
    assert set(env.observation_space.spaces) == {"robot0_eef_pos"}


def test_spaces_without_gymnasium_fail_actionably(monkeypatch):
    monkeypatch.setitem(sys.modules, "gymnasium", None)
    env = RemoteEnv("http://127.0.0.1:1", "token")
    env._refresh_env(ENV)
    with pytest.raises(ImportError, match=r"world-engine-client\[gym\]"):
        env.action_space  # noqa: B018
