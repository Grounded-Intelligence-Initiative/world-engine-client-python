"""Gymnasium spaces derived from the server's negotiated env blob.

Gymnasium is imported inside :func:`build_spaces` so the base package stays
numpy-only; ``RemoteEnv.action_space`` and the Gymnasium adapter share this.
"""

import numpy as np


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
    low[-1], high[-1] = -1, 1
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
