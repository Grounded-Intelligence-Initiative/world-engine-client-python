"""Gymnasium spaces derived from the server's negotiated env blob.

Gymnasium is imported inside :func:`build_spaces` so the base package stays
numpy-only; ``RemoteEnv.action_space`` and the Gymnasium adapter share this.
"""

import numpy as np


def gripper_dims(schema: dict, action_dim: int) -> list[int]:
    """The action dims that are gripper commands, by the server's rule: the
    schema's ``gripper_indices`` (one per robot in a scene with several), else
    its ``gripper_index``, else the last dim (a server from before either).
    The server keeps them within ±1 in every control mode, so the action
    space does too."""
    indices = schema.get("gripper_indices")
    if indices:
        return [int(i) for i in indices]
    if schema.get("gripper_index") is not None:
        return [int(schema["gripper_index"])]
    return [int(action_dim) - 1]


def build_spaces(metadata: dict, control_mode: str, action_dim: int):
    """→ ``(action_space, observation_space)`` for the active control mode."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("Install world-engine-client[gym] to use Gymnasium spaces") from exc

    schema = metadata["control_mode_schemas"][control_mode]
    lo, hi = schema["bounds"] or (-np.inf, np.inf)
    low = np.full(action_dim, lo, np.float32)
    high = np.full(action_dim, hi, np.float32)
    for i in gripper_dims(schema, action_dim):
        low[i], high[i] = -1, 1
    spaces = {
        k: gym.spaces.Box(-np.inf, np.inf, tuple(v["shape"]), np.float32)
        for k, v in metadata["observation_schema"].items()
    }
    cameras = metadata.get("cameras", {})
    if cameras:
        spaces["rgb"] = gym.spaces.Dict(
            {
                k: gym.spaces.Box(0, 255, (v["height"], v["width"], 3), np.uint8)
                for k, v in cameras.items()
            }
        )
    return gym.spaces.Box(low, high), gym.spaces.Dict(spaces)
