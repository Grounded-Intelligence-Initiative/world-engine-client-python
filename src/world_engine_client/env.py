"""RemoteEnv: a Gymnasium-shaped client for the World Engine /control API.

Synchronous by design — an inference loop is synchronous. Transport is plain
HTTP POST (stdlib urllib): one round-trip per step, which the lockstep
semantics require anyway. Obs arrays come back as JSON lists and are converted
to numpy here; camera images are a future obs key (see the design doc §4).

Proto 2 (server ≥ the editor-teleop release): the default control mode is
``osc_pose_delta`` — a 7-dim action ``[dpos(3), drot_axisangle(3), gripper]``
in [-1, 1] (robosuite OSC_POSE conventions; dpos scales by ``ee_step`` m,
drot by ``ee_rot_step`` rad, gripper is a sign-driven rate). Obs use robosuite
naming (``robot0_eef_pos``, ``robot0_eef_quat`` in scalar-last **xyzw**,
``robot0_joint_pos`` …). Control mode + obs keys are negotiated at
:func:`connect` or via :meth:`RemoteEnv.configure`; switches land at episode
boundaries (immediately before the first reset, else pending until the next).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np

from .errors import WorldEngineClientError, raise_for

_PROTO = 2


def connect(
    token: str,
    url: str = "http://127.0.0.1:8080",
    *,
    takeover: bool = False,
    timeout: float = 30.0,
    control_mode: str | None = None,
    obs_keys: list[str] | None = None,
) -> RemoteEnv:
    """Bind to the server as THE controller and return a ready RemoteEnv.

    ``token`` is printed by ``world-engine serve --allow-remote-control`` (or
    shown in the editor Inspector's Python API card during a Simulate session).
    ``takeover=True`` revokes a controller that died without releasing.
    ``control_mode`` / ``obs_keys`` negotiate the space at connect time (see
    ``env.metadata['supported_control_modes']`` for what the server offers)."""
    env = RemoteEnv(url.rstrip("/"), token, timeout=timeout)
    env._connect(takeover=takeover, control_mode=control_mode, obs_keys=obs_keys)
    return env


class RemoteEnv:
    """reset/step/close over HTTP. Use via :func:`connect`."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0):
        self._base = base_url
        self._token = token
        self._timeout = timeout
        self._session_id: str | None = None
        self.metadata: dict = {}
        self.action_dim: int = 7
        self.observation_keys: list[str] = []
        self.control_mode: str | None = None

    # ----- Gym API ------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        reply = self._post("/control/reset", {"seed": seed})
        self._refresh_env(reply.get("env"))
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

    def configure(
        self, *, control_mode: str | None = None, obs_keys: list[str] | None = None
    ) -> dict:
        """Negotiate the control mode / obs keys (proto 2).

        Applies immediately when no episode is active; mid-episode it is
        recorded as PENDING and lands at the next :meth:`reset` (spaces never
        change under a running episode). Returns the server's resolved env
        blob; ``action_dim``/``observation_keys``/``control_mode`` refresh once
        the switch is live (i.e. now, or after the next reset)."""
        body: dict = {}
        if control_mode is not None:
            body["control_mode"] = control_mode
        if obs_keys is not None:
            body["obs_keys"] = list(obs_keys)
        reply = self._post("/control/configure", body)
        if not reply.get("pending"):
            self._refresh_env(reply.get("env"))
        return reply.get("env", {})

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
    def _connect(
        self,
        *,
        takeover: bool,
        control_mode: str | None = None,
        obs_keys: list[str] | None = None,
    ) -> None:
        body: dict = {"proto": _PROTO, "takeover": takeover}
        if control_mode is not None:
            body["control_mode"] = control_mode
        if obs_keys is not None:
            body["obs_keys"] = list(obs_keys)
        reply = self._post("/control/connect", body, with_session=False)
        self._session_id = reply["session_id"]
        self._refresh_env(reply.get("env"))

    def _refresh_env(self, env: dict | None) -> None:
        """Track the server's resolved env blob (action space follows the
        active control mode; a proto-1 server just reports its fixed dims)."""
        if not env:
            return
        self.metadata = env
        self.action_dim = int(env.get("action_dim", self.action_dim))
        self.observation_keys = list(env.get("observation_keys", self.observation_keys))
        self.control_mode = env.get("active_control_mode", self.control_mode)

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
                    "Is `world-engine serve --mode simulator --allow-remote-control` "
                    "running (or an editor Simulate session live)?",
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
