"""Gymnasium spaces follow the negotiated control mode, obs schema and cameras."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import ENV

from world_engine_client import RemoteEnv
from world_engine_client.gym_adapter import RemoteGymEnv
from world_engine_client.spaces import build_spaces


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
