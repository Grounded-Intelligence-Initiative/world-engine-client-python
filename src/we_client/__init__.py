"""world-engine (import name ``we_client``): drive a remote World Engine sim
with Gym semantics.

    import we_client
    env = we_client.connect("<control token>", url="http://<host>:8080")
    obs, info = env.reset(seed=0)
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step([0.1, 0, -0.2, -1.0])
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

The token is printed by ``world-engine serve --mode simulator
--allow-remote-control``. Every browser viewing that server's viewer.html sees
your episode live. Stdlib HTTP + numpy only — no gymnasium requirement (the
API is 5-tuple compatible).
"""

from .env import RemoteEnv, connect
from .errors import (
    ControllerConflictError,
    ProtocolError,
    ResetRequiredError,
    TokenError,
    WorldEngineClientError,
)

__all__ = [
    "connect",
    "RemoteEnv",
    "WorldEngineClientError",
    "TokenError",
    "ControllerConflictError",
    "ResetRequiredError",
    "ProtocolError",
]
__version__ = "0.1.0"
