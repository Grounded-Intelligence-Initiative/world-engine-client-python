"""RemoteEnv: a Gymnasium-shaped client for the World Engine /control API.

Synchronous by design — an inference loop is synchronous. Transport is plain
HTTP POST (persistent stdlib http.client): one round-trip per step, which the lockstep
semantics require anyway. Obs arrays come back as JSON lists and are converted
to numpy here. Camera images are rendered on this machine (see render.py): the
server sends the poses they are drawn from with every observation.

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

import http.client
import json
import math
import select
import threading
import time
import urllib.parse
import warnings

import numpy as np

from .errors import (
    ProtocolError,
    RecoveryRequiredError,
    RendererError,
    RenderFetchError,
    WorldEngineClientError,
    raise_for,
)

MAX_RESPONSE = 64 * 1024 * 1024

_PROTO = 3
# Seconds between heartbeats. The server drops a proto-3 controller that sends
# nothing for 90 s (its lease).
_HEARTBEAT_SECONDS = 30.0


def connect(
    token: str,
    url: str = "http://127.0.0.1:8080",
    *,
    takeover: bool = False,
    timeout: float = 30.0,
    control_mode: str | None = None,
    obs_keys: list[str] | None = None,
    mode: str = "reset",
    images: bool = False,
    cameras: list[str] | None = None,
    expected_instance_id: str | None = None,
    proto: int = _PROTO,
    access_provider=None,
    render: str = "auto",
    render_options: dict | None = None,
    normalize_reward: bool = True,
) -> RemoteEnv:
    """Bind to the server as THE controller and return a ready RemoteEnv.

    ``token`` is printed by ``world-engine serve --allow-remote-control`` (or
    shown in the editor Inspector's Python API card during a Simulate session).
    ``takeover=True`` revokes a controller that died without releasing.
    ``control_mode`` / ``obs_keys`` negotiate the space at connect time (see
    ``env.metadata['supported_control_modes']`` for what the server offers).

    Camera images (``images=True`` or ``cameras=[...]``) are rendered on this
    machine; the server only simulates. That needs the ``render`` extra. A
    server with renderer D (render protocol 2) has them drawn by
    :class:`world_engine_client.render_wgsl.WGSLRenderer` (wgpu, no browser); an
    older one by headless Chrome (:class:`world_engine_client.render.LocalRenderer`),
    and so does a machine where no GPU or software adapter renders for D but
    Chrome does. Without a hardware GPU they are rendered in software, with a
    RuntimeWarning. ``render`` accepts ``"auto"`` or ``"local"`` (the same
    thing; kept for callers that pass it). ``render_options``: ``renderer``
    (``"auto"``, ``"wgsl"`` or ``"chrome"``; the WE_RENDERER environment
    variable sets the default), ``cache_dir``, ``scene_name``,
    ``allow_software`` (default True; False refuses software rendering),
    ``timeout``; for D ``adapter`` (``"auto"``, ``"low-power"``, ``"software"``
    or ``"name:<part of the adapter name>"``; default WE_RENDER_ADAPTER) and
    ``color_format``; for Chrome ``chrome_args`` and ``auto_install``. Scene
    assets are kept in the render cache (:mod:`world_engine_client.cache`): the
    hosting platform's arrive encrypted and stay encrypted on disk.

    ``normalize_reward`` (default True) returns each reward as the server sends
    it. The server divides a ``staged`` reward by the task's total points, so an
    episode that finishes sums to 1.0. With False a staged reward comes back in
    points, ``reward * info["task_progress"]["reward"]["max"]``: the points the
    step earned, less the failure penalty on the step that fails. A step that
    gives partial credit back is negative without a failure. Over an episode
    from :meth:`RemoteEnv.reset` they add up to ``earned - penalty`` of the last
    ``info["task_progress"]["reward"]``. ``sparse`` and ``dense`` rewards, as
    the env's ``reward_type`` names them, come back unchanged either way."""
    if render not in ("auto", "local"):
        raise ValueError("render must be 'auto' or 'local': servers no longer render camera images")
    if not isinstance(normalize_reward, bool):
        raise TypeError(f"normalize_reward must be True or False, not {normalize_reward!r}")
    env = RemoteEnv(url.rstrip("/"), token, timeout=timeout)
    env.normalize_reward = normalize_reward
    env._access_provider = access_provider
    env._render_request = render
    env._render_options = dict(render_options or {})
    try:
        env._connect(
            takeover=takeover,
            control_mode=control_mode,
            obs_keys=obs_keys,
            mode=mode,
            images=images,
            cameras=cameras,
            expected_instance_id=expected_instance_id,
            proto=proto,
        )
    except BaseException:
        env.close()
        raise
    return env


class RemoteEnv:
    """reset/step/close over HTTP. Use via :func:`connect`."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 30.0):
        self._base = base_url
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("url must be an HTTP(S) endpoint without embedded credentials")
        transport = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        self._http = transport(parsed.hostname, parsed.port, timeout=timeout)
        self._prefix = parsed.path.rstrip("/")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._heartbeat_thread = None
        self._uncertain = False
        self._latest = None
        self._episode_id = None
        self._access_provider = None
        self._heartbeat_error = None
        self._token = token
        self._timeout = timeout
        self._session_id: str | None = None
        self.metadata: dict = {}
        self.action_dim: int = 7
        self.observation_keys: list[str] = []
        self.control_mode: str | None = None
        self._render_request = "server"
        self._render_options: dict = {}
        self._render_split: tuple[dict, dict, str] = ({}, {}, "auto")
        self.renderer = None  # LocalRenderer when images are rendered on this machine
        # False: step() returns a staged reward in points (see connect).
        self.normalize_reward = True

    # ----- Gym API ------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        reply = self._post("/control/reset", {"seed": seed, "options": options})
        self._uncertain = False
        if (
            self._heartbeat_thread is not None
            and not self._heartbeat_thread.is_alive()
            and not self._stop.is_set()
        ):
            self._heartbeat_error = None
            self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._heartbeat_thread.start()
        self._refresh_env(reply.get("env"))
        return self._remember(reply), reply.get("info", {})

    def step(self, action):
        if self._uncertain:
            raise RecoveryRequiredError(
                "previous action outcome is uncertain; explicitly reset or reconnect"
            )
        a = np.asarray(action)
        if a.shape != (self.action_dim,) or a.dtype.kind not in "iuf" or not np.isfinite(a).all():
            raise ValueError(
                f"action must be a finite numeric vector of shape ({self.action_dim},)"
            )
        reply = self._post("/control/step", {"action": [float(x) for x in a]})
        info = reply.get("info", {})
        return (
            self._remember(reply),
            self._reward(float(reply["reward"]), info),
            bool(reply["terminated"]),
            bool(reply["truncated"]),
            info,
        )

    def _reward(self, reward: float, info: dict) -> float:
        """The step's reward, in points for a staged task when normalize_reward
        is False. The server divides a staged reward by the task's total
        points, which every step's info carries; sparse and dense rewards are
        not divided."""
        if self.normalize_reward:
            return reward
        progress = info.get("task_progress")
        totals = progress.get("reward") if isinstance(progress, dict) else None
        kind = self.metadata.get("reward_type")
        if kind is None:
            # A server older than staged rewards, or no task: only a staged
            # reward has totals in the step's info.
            staged = totals is not None
        else:
            staged = kind == "staged"
        if not staged:
            return reward
        total = totals.get("max") if isinstance(totals, dict) else None
        points = reward * total if type(total) in (int, float) and total > 0 else math.nan
        if not math.isfinite(points):
            # The step ran, but its reward cannot be given in points.
            self._uncertain = True
            raise ProtocolError(
                "cannot give a staged reward in points: info['task_progress']['reward']['max'] "
                "is missing or not a positive number; explicit reset required",
                fixes=["connect with normalize_reward=True to take rewards as the server sends"],
            )
        return points

    def observe(self):
        reply = self._post("/control/observe", {})
        return self._remember(reply), reply.get("info", {})

    def render(self, camera=None):
        if self._latest is None:
            raise WorldEngineClientError("observe or reset before render")
        images = self._latest.get("rgb", {})
        if not images:
            raise WorldEngineClientError("connect with images=True to enable RGB rendering")
        return images[camera or next(iter(images))]

    @property
    def action_space(self):
        """Gymnasium Box for the active control mode (needs the ``gym`` extra)."""
        return self._spaces()[0]

    @property
    def observation_space(self):
        """Gymnasium Dict of numeric observations (+ ``rgb`` when images are on)."""
        return self._spaces()[1]

    def _spaces(self):
        from .spaces import build_spaces

        return build_spaces(self.metadata, self.control_mode, self.action_dim)

    def as_gymnasium(self):
        try:
            from .gym_adapter import RemoteGymEnv
        except ImportError as exc:
            raise ImportError("Install world-engine-client[gym] to use Gymnasium") from exc
        return RemoteGymEnv(self)

    def _remember(self, reply):
        self.latest_info = reply.get("info", {})
        self._episode_id = self.latest_info.get("episode_id", self._episode_id)
        self._latest = self._np_obs(reply["obs"])
        return self._latest

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
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self._timeout + 1)
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._session_id is None:
            self._http.close()
            return
        try:
            self._post("/control/release", {})
        except Exception:
            pass  # already revoked / expired platform credential / server gone
        self._session_id = None
        self._http.close()

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
        mode="reset",
        images=False,
        cameras=None,
        expected_instance_id=None,
        proto=_PROTO,
    ) -> None:
        local = bool(images or cameras)
        dac1_ok = d_ok = chrome_ok = False
        if local:
            from . import render as local_render
            from . import render_wgsl

            self._render_split = render_wgsl.env_options(self._render_options)
            wants = self._render_split[2]
            # Renderer D needs wgpu; the browser path Playwright and websockets.
            # The render extra brings both, and cryptography for the platform's
            # encrypted scene assets. A local world-engine serve sends its scene
            # plain, so a session with it works without cryptography.
            d_ok = wants != "chrome" and render_wgsl.available()
            chrome_ok = wants != "wgsl" and local_render._available_plain()
            dac1_ok = local_render.crypto_available() and (d_ok or chrome_ok)
            platform = self._access_provider is not None
            if not (d_ok or chrome_ok) or (platform and not dac1_ok):
                raise local_render.RendererError(
                    "camera images are rendered on this machine and need the render extra",
                    code="renderer_unavailable",
                    fixes=local_render._INSTALL_FIXES,
                )
        body: dict = {
            "proto": proto,
            "takeover": takeover,
            "mode": mode,
            "images": images,
            "cameras": cameras,
            "expected_instance_id": expected_instance_id,
        }
        if control_mode is not None:
            body["control_mode"] = control_mode
        if obs_keys is not None:
            body["obs_keys"] = list(obs_keys)
        if local:
            body["render"] = "client"
            # This client reads the platform's encrypted scene assets (DAC1)
            # when cryptography is installed; a hosted engine refuses image
            # sessions of clients that do not.
            if dac1_ok:
                body["asset_encryption"] = ["dac1"]
            # Render protocols: 1 is the browser path (the engine's capture
            # page), 2 renderer D with a pipeline version this package ships.
            # An engine answers 2 only when it supports one of the pipelines
            # and can describe the scene in it; an older engine ignores these
            # fields and answers 1 (design doc 16 §9.3).
            body["render_protocols"] = [p for p, ok in ((1, chrome_ok), (2, d_ok)) if ok]
            if d_ok:
                body["pipelines"] = [render_wgsl.PIPELINE]
        reply = self._post("/control/connect", body, with_session=False)
        if reply.get("proto") != proto:
            raise ProtocolError("server did not honor the requested protocol")
        if expected_instance_id and reply.get("instance_id") != expected_instance_id:
            raise ProtocolError("server instance identity mismatch")
        self._session_id = reply["session_id"]
        if proto == 3:
            self._episode_id = reply.get("episode_id")
        self._refresh_env(reply.get("env"))
        bundle = reply.get("render") if local else None
        if local:
            if not isinstance(bundle, dict) or not bundle:
                raise ProtocolError(
                    "this server predates client rendering; upgrade it, or connect with "
                    "images=False"
                )
            protocol = bundle.get("protocol")
            if protocol not in (1, 2) or (protocol == 2 and not d_ok):
                raise ProtocolError(
                    f"the server answered render protocol {protocol!r}, which this client did "
                    "not offer",
                    fixes=["pip install -U 'world-engine-client[render]'"],
                )
            pipeline = (bundle.get("pipeline") or {}).get("version")
            if protocol == 2 and pipeline != render_wgsl.PIPELINE:
                raise ProtocolError(
                    f"the server chose pipeline {pipeline!r}; this client ships "
                    f"{render_wgsl.PIPELINE}",
                    fixes=["pip install -U 'world-engine-client[render]'"],
                )
            if protocol == 1 and not chrome_ok:
                why = (
                    "render_options asks for renderer 'wgsl'"
                    if self._render_split[2] == "wgsl"
                    else "the browser path needs Playwright and websockets"
                )
                raise local_render.RendererError(
                    "this server offers only the browser path (render protocol 1, an engine "
                    f"from before renderer D), and {why}",
                    code="renderer_unavailable",
                    fixes=[
                        "upgrade the server to an engine with renderer D",
                        "or pip install playwright websockets (the browser path)",
                        "or pass images=False for a state-only session",
                    ],
                )
        if proto == 3:
            # Start before the renderer: its start can take longer than the
            # 90 s lease (a large scene rendered in software). connect() stops
            # the thread if the start fails.
            self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._heartbeat_thread.start()
        if local:
            self.renderer = self._start_renderer(bundle, chrome_ok)

    def _start_renderer(self, bundle: dict, chrome_ok: bool):
        """Renderer D for protocol 2, the browser path for protocol 1; the
        browser path also when D finds no adapter that renders and Chrome is
        there (design doc 16 §9.9 runtime, step 5)."""
        from .render import LocalRenderer
        from .render_wgsl import NoRenderAdapter, WGSLRenderer

        d_opts, chrome_opts, _ = self._render_split
        parsed = urllib.parse.urlsplit(self._base)
        common = {
            "bundle": bundle,
            "platform": self._access_provider is not None,
            "origin": f"{parsed.scheme}://{parsed.netloc}",
            "scene_id": self.metadata.get("scene_id"),
        }
        args = (self._open_render, bundle["scene_init"], bundle["digests"])
        d_failure = None
        if bundle.get("protocol") == 2:
            try:
                return WGSLRenderer(*args, **common, **d_opts)
            except NoRenderAdapter as exc:
                if not chrome_ok:
                    raise
                d_failure = exc
                warnings.warn(
                    "renderer D found no GPU or software adapter that renders here, so camera "
                    f"images come from headless Chrome instead, which is slower ({exc.args[0]})",
                    RuntimeWarning,
                    stacklevel=4,
                )
        try:
            return LocalRenderer(*args, **common, **chrome_opts)
        except RendererError as exc:
            if d_failure is None:
                raise
            raise NoRenderAdapter(
                f"nothing renders camera images here: renderer D: {d_failure.args[0]} | "
                f"headless Chrome: {exc.args[0]}",
                code="renderer_unavailable",
            ) from exc

    def _heartbeat(self):
        while not self._stop.wait(_HEARTBEAT_SECONDS):
            try:
                self._post("/control/heartbeat", {})
            except Exception as exc:
                self._heartbeat_error = exc
                self._uncertain = True
                return

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
        # Serialize heartbeat, credential refresh and actions on one persistent connection.
        # Never retry a request: the server might already have executed the action.
        with self._lock:
            body = dict(payload)
            if path == "/control/step" and self._episode_id:
                body["expected_episode_id"] = self._episode_id
            if with_session:
                body["session_id"] = self._session_id
            headers = {"Content-Type": "application/json", "X-We-Control-Token": self._token}
            if self._access_provider:
                headers["Authorization"] = "Bearer " + self._access_provider()
            encoded = json.dumps(body, allow_nan=False).encode()
            started = time.perf_counter()
            self._drop_closed_connection()
            try:
                self._http.request("POST", self._prefix + path, body=encoded, headers=headers)
                response = self._http.getresponse()
                data = response.read(MAX_RESPONSE + 1)
                if len(data) > MAX_RESPONSE:
                    raise ProtocolError("response exceeds size limit")
                received = time.perf_counter()
                reply = json.loads(data)
                if not isinstance(reply, dict) or "ok" not in reply:
                    raise ProtocolError("invalid response envelope")
            except (OSError, http.client.HTTPException, ValueError, ProtocolError) as exc:
                self._http.close()
                if path in ("/control/step", "/control/reset"):
                    self._uncertain = True
                    raise RecoveryRequiredError(
                        "request outcome uncertain; explicitly reset or reconnect"
                    ) from exc
                raise ProtocolError(f"request failed at {path}: {exc}") from exc
            if reply.get("ok") and path in ("/control/reset", "/control/step", "/control/observe"):
                try:
                    if self.renderer is not None and "render_state" not in reply:
                        raise ValueError("observation without the poses to render")
                    if not isinstance(reply["obs"], dict) or not isinstance(reply["info"], dict):
                        raise ValueError("invalid observation or info")
                    for key, spec in (
                        reply.get("env", self.metadata).get("observation_schema", {}).items()
                    ):
                        value = np.asarray(reply["obs"][key])
                        if (
                            value.shape != tuple(spec["shape"])
                            or value.dtype.kind not in "iuf"
                            or not np.isfinite(value).all()
                            or (np.abs(value) > np.finfo(np.float32).max).any()
                        ):
                            raise ValueError(f"invalid observation: {key}")
                    if path == "/control/step" and (
                        type(reply["terminated"]) is not bool
                        or type(reply["truncated"]) is not bool
                        or not np.isfinite(float(reply["reward"]))
                    ):
                        raise ValueError("invalid step result")
                except (KeyError, ValueError, TypeError) as exc:
                    if path in ("/control/step", "/control/reset"):
                        self._uncertain = True
                    raise ProtocolError(
                        "malformed observation result; explicit reset may be required"
                    ) from exc
            if not reply.get("ok"):
                if reply.get("code") == "recovery_required":
                    self._uncertain = True
                raise_for(reply)
        if path not in ("/control/reset", "/control/step", "/control/observe"):
            return reply
        server_ms = reply["info"].get("timings", {}).get("server_ms", 0)
        timings = {
            "server_ms": server_ms,
            "transport_ms": max(0, (received - started) * 1000 - server_ms),
            "response_bytes": len(data),
        }
        if self.renderer is not None:
            # Outside the lock: the heartbeat must not wait for a frame. A
            # render failure leaves physics as stepped; observe() renders again.
            rendering = time.perf_counter()
            reply["obs"]["rgb"] = self.renderer.render(reply["render_state"])
            timings["render_ms"] = (time.perf_counter() - rendering) * 1000
            # Which renderer and adapter drew the images (and, for a software
            # adapter, why no GPU did).
            render_info = getattr(self.renderer, "info", None)
            if isinstance(render_info, dict):
                reply["info"]["render"] = dict(render_info)
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        self.last_timings = timings
        return reply

    def _drop_closed_connection(self) -> None:
        """Servers close keep-alive connections that stay idle (Node after
        5 s), for instance while the renderer starts. A request sent on such
        a socket is lost, and for a step or reset the client could not tell
        whether it ran. Between requests nothing may arrive on the socket,
        so a readable one has been closed by the server (or holds garbage):
        open a new connection before sending. Nothing has been sent yet, so
        this is no retry."""
        sock = self._http.sock
        if sock is None:
            return
        try:
            readable, _, _ = select.select([sock], [], [], 0)
        except (OSError, ValueError):
            readable = [sock]
        if readable:
            self._http.close()

    def _open_render(self, path: str, headers: dict | None = None):
        """GET <server>/render<path> with this session's credentials, on its own
        connection (the renderer runs on another thread). ``headers`` adds
        request headers, a ``Range`` to resume a download. 404 raises
        FileNotFoundError, any other error status RenderFetchError with the
        server's code and message. Waits at most 60 s for each piece of data."""
        parsed = urllib.parse.urlsplit(self._base)
        transport = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        conn = transport(parsed.hostname, parsed.port, timeout=max(self._timeout, 60))
        sent = {"X-We-Control-Token": self._token}
        if self._access_provider:
            sent["Authorization"] = "Bearer " + self._access_provider()
        sent.update(headers or {})
        try:
            conn.request("GET", self._prefix + "/render" + path, headers=sent)
            response = conn.getresponse()
        except BaseException:
            conn.close()
            raise
        if response.status == 404:
            conn.close()
            raise FileNotFoundError(path)
        if response.status not in (200, 206):
            detail = response.read(4096)
            conn.close()
            try:
                reply = json.loads(detail)
            except ValueError:
                reply = None
            if not isinstance(reply, dict):
                reply = {"error": detail[:500].decode("utf-8", "replace")}
            fixes = reply.get("fixes")
            raise RenderFetchError(
                response.status,
                path,
                code=reply.get("code") if isinstance(reply.get("code"), str) else None,
                message=str(reply.get("error") or ""),
                fixes=fixes if isinstance(fixes, list) else None,
            )
        return _Owned(response, conn)

    @staticmethod
    def _np_obs(obs: dict) -> dict:
        out: dict = {}
        for k, v in obs.items():
            if (
                isinstance(v, list)
                and k != "object_ids"
                and (not v or isinstance(v[0], (int, float)))
            ):
                out[k] = np.asarray(v, dtype=np.float32)
            else:
                out[k] = v
        return out


class _Owned:
    """A response that closes its connection when the caller is done."""

    def __init__(self, response, conn):
        self._response, self._conn = response, conn
        self.status = response.status

    def read(self, *a):
        return self._response.read(*a)

    def getheader(self, name, default=None):
        return self._response.getheader(name, default)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._conn.close()
