"""RemoteEnv: a Gymnasium-shaped client for the World Engine /control API.

Synchronous by design — an inference loop is synchronous. Transport is plain
HTTP POST (stdlib urllib): one round-trip per step, which the lockstep
semantics require anyway. Obs arrays come back as JSON lists and are converted
to numpy here; camera images are a future obs key (see the design doc §4).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np

from .errors import WorldEngineClientError, raise_for

_PROTO = 1


def connect(
    token: str,
    url: str = "http://127.0.0.1:8080",
    *,
    takeover: bool = False,
    timeout: float = 30.0,
) -> RemoteEnv:
    """Bind to the server as THE controller and return a ready RemoteEnv.

    ``token`` is printed by ``world-engine serve --allow-remote-control``.
    ``takeover=True`` revokes a controller that died without releasing."""
    env = RemoteEnv(url.rstrip("/"), token, timeout=timeout)
    env._connect(takeover=takeover)
    return env


class RemoteEnv:
    """reset/step/close over HTTP. Use via :func:`connect`."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0):
        self._base = base_url
        self._token = token
        self._timeout = timeout
        self._session_id: str | None = None
        self.metadata: dict = {}
        self.action_dim: int = 4
        self.observation_keys: list[str] = []

    # ----- Gym API ------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        reply = self._post("/control/reset", {"seed": seed})
        return self._np_obs(reply["obs"]), reply.get("info", {})

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        reply = self._post("/control/step", {"action": [float(x) for x in a]})
        return (
            self._np_obs(reply["obs"]),
            float(reply["reward"]),
            bool(reply["terminated"]),
            bool(reply["truncated"]),
            reply.get("info", {}),
        )

    def close(self) -> None:
        if self._session_id is None:
            return
        try:
            self._post("/control/release", {})
        except WorldEngineClientError:
            pass  # already revoked / server gone — nothing to release
        self._session_id = None

    def __enter__(self) -> RemoteEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----- wire ----------------------------------------------------------------#
    def _connect(self, *, takeover: bool) -> None:
        reply = self._post(
            "/control/connect", {"proto": _PROTO, "takeover": takeover}, with_session=False
        )
        self._session_id = reply["session_id"]
        self.metadata = reply.get("env", {})
        self.action_dim = int(self.metadata.get("action_dim", 4))
        self.observation_keys = list(self.metadata.get("observation_keys", []))

    def _post(self, path: str, payload: dict, *, with_session: bool = True) -> dict:
        body = dict(payload)
        if with_session:
            body["session_id"] = self._session_id
        req = urllib.request.Request(
            self._base + path,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-We-Control-Token": self._token,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                reply = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                reply = json.loads(e.read().decode("utf-8"))
            except ValueError:
                raise WorldEngineClientError(f"HTTP {e.code} from {path}") from e
        except (urllib.error.URLError, OSError) as e:
            raise WorldEngineClientError(
                f"cannot reach the World Engine server at {self._base}",
                fixes=[
                    "Is `world-engine serve --mode simulator --allow-remote-control` running?",
                    "Check the url= argument (host/port, and any SSH -L / Tailscale routing).",
                ],
            ) from e
        if not reply.get("ok"):
            raise_for(reply)
        return reply

    @staticmethod
    def _np_obs(obs: dict) -> dict:
        out: dict = {}
        for k, v in obs.items():
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                out[k] = np.asarray(v, dtype=np.float32)
            else:
                out[k] = v
        return out
