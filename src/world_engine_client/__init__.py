"""world-engine-client (import name ``world_engine_client``): drive a remote
World Engine sim with Gym semantics.

    import world_engine_client as we_client
    env = we_client.connect("<control token>", url="http://<host>:8080")
    obs, info = env.reset(seed=0)
    for _ in range(100):
        # 7-dim [dpos(3), drot_axisangle(3), gripper] in [-1,1]
        obs, reward, terminated, truncated, info = env.step([0.1, 0, -0.2, 0, 0, 0, -1])
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

The token is printed by ``world-engine serve --mode simulator
--allow-remote-control`` — or shown in the editor Inspector's Python API card
while a Simulate session runs (you then drive the exact scene the editor user
is looking at). Every browser viewer sees your episode live. Stdlib HTTP +
numpy only — no gymnasium requirement (the API is 5-tuple compatible).
"""

from .env import RemoteEnv, connect
from .errors import (
    BadConfigError,
    ControllerConflictError,
    ProtocolError,
    RecordingPendingError,
    RecoveryRequiredError,
    ResetRequiredError,
    TeleopActiveError,
    TokenError,
    WorldEngineClientError,
)
from .render import RendererError
from .rotations import (
    axisangle_to_matrix,
    matrix_to_axisangle,
    matrix_to_rot6d,
    rot6d_to_matrix,
)

__all__ = [
    "connect",
    "RemoteEnv",
    "WorldEngineClientError",
    "TokenError",
    "ControllerConflictError",
    "ResetRequiredError",
    "ProtocolError",
    "TeleopActiveError",
    "BadConfigError",
    "RecoveryRequiredError",
    "RecordingPendingError",
    "RendererError",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "axisangle_to_matrix",
    "matrix_to_axisangle",
]
__version__ = "0.6.0"
