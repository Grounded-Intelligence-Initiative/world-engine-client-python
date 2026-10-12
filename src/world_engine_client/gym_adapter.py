"""Optional Gymnasium adapter; raw compatibility strings live in info."""

import gymnasium as gym

from .spaces import build_spaces


class RemoteGymEnv(gym.Env):
    render_mode = "rgb_array"

    def __init__(self, remote):
        self.remote = remote
        cameras = remote.metadata.get("cameras", {})
        self.render_mode = "rgb_array" if cameras else None
        self.metadata = {
            "render_modes": ["rgb_array"] if cameras else [],
            "render_fps": round(1 / remote.metadata["control_interval"]),
        }
        self.action_space, self.observation_space = build_spaces(
            remote.metadata, remote.control_mode, remote.action_dim
        )

    @staticmethod
    def _adapt(obs, info):
        obs = dict(obs)
        # Transport identities change on seeded resets. They remain available on
        # remote.latest_info; Gym info contains episode-deterministic task data.
        info = {k: v for k, v in info.items() if k not in ("episode_id", "instance_id", "timings")}
        return {k: v for k, v in obs.items() if k != "object_ids"}, {
            **info,
            "object_ids": obs.get("object_ids", []),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._adapt(*self.remote.reset(seed=seed, options=options))

    def step(self, action):
        obs, reward, term, trunc, info = self.remote.step(action)
        obs, info = self._adapt(obs, info)
        return obs, reward, term, trunc, info

    def render(self):
        return self.remote.render() if self.render_mode else None

    def close(self):
        self.remote.close()
