"""RemoteEnv transport behaviour against a programmable stub server.

The rule under test: a request whose outcome is unknown is never retried, and
step/reset failures block further actions until an explicit reset or reconnect.
"""

import threading

import numpy as np
import pytest
from conftest import DROP, ENV, OBS

import world_engine_client as wec
from world_engine_client import env as env_module

CAMERAS = {"front": {"width": 8, "height": 6}}


@pytest.fixture
def env(stub):
    remote = wec.connect("secret", url=stub.url, timeout=5)
    yield remote
    remote.close()


def test_connect_negotiates_protocol_and_close_releases(stub):
    remote = wec.connect(
        "secret",
        url=stub.url + "/",
        mode="resume",
        expected_instance_id="inst",
        control_mode="ee_pos_delta",
        obs_keys=["robot0_eef_pos"],
    )
    path, body, headers = stub.calls[0]
    assert path == "/control/connect" and "session_id" not in body
    assert body["proto"] == 3 and body["mode"] == "resume" and body["takeover"] is False
    assert body["images"] is False and body["cameras"] is None
    assert body["expected_instance_id"] == "inst"
    assert body["control_mode"] == "ee_pos_delta" and body["obs_keys"] == ["robot0_eef_pos"]
    assert headers["X-We-Control-Token"] == "secret" and "Authorization" not in headers
    assert remote.action_dim == 4 and remote.control_mode == "ee_pos_delta"
    assert remote.metadata["instance_id"] == "inst"
    remote.close()
    assert stub.calls[-1][0] == "/control/release"
    assert stub.calls[-1][1]["session_id"] == "sess"
    remote.close()  # idempotent: no second release
    assert stub.paths().count("/control/release") == 1


def test_url_prefix_is_preserved_and_credentials_rejected(stub):
    remote = wec.connect("secret", url=stub.url + "/api/v1/sessions/abc")
    assert stub.calls[0][0] == "/api/v1/sessions/abc/control/connect"
    remote.close()
    assert stub.calls[-1][0] == "/api/v1/sessions/abc/control/release"
    for bad in ("ftp://host", "http://user:pw@host", "host:8080", "http://"):
        with pytest.raises(ValueError, match="HTTP"):
            wec.RemoteEnv(bad, "token")


def test_step_carries_session_and_episode_identity(env, stub):
    obs, info = env.reset(seed=7)
    assert stub.calls[-1][1] == {"seed": 7, "options": None, "session_id": "sess"}
    assert isinstance(obs["robot0_eef_pos"], np.ndarray)
    assert obs["robot0_eef_pos"].dtype == np.float32 and obs["object_ids"] == ["cube"]
    obs, reward, terminated, truncated, info = env.step([0.5, 0, 0, -1])
    body = stub.calls[-1][1]
    assert body["action"] == [0.5, 0.0, 0.0, -1.0]
    assert body["expected_episode_id"] == "e1" and body["session_id"] == "sess"
    assert (reward, terminated, truncated) == (0.5, False, False)
    assert type(reward) is float and type(terminated) is bool
    assert info["step_id"] == 1 and env.latest_info is info


def test_access_provider_is_consulted_for_every_request(stub):
    tokens = iter(f"access-{i}" for i in range(100))
    remote = wec.connect("secret", url=stub.url, access_provider=lambda: next(tokens))
    remote.observe()
    assert [c[2]["Authorization"] for c in stub.calls] == ["Bearer access-0", "Bearer access-1"]
    remote.close()


@pytest.mark.parametrize(
    "action",
    [[0, 0, 0], np.zeros((2, 2)), [0, 0, 0, np.nan], [0, 0, 0, np.inf], ["a", "b", "c", "d"]],
    ids=["short", "matrix", "nan", "inf", "strings"],
)
def test_invalid_actions_never_leave_the_client(env, stub, action):
    env.reset()
    sent = len(stub.calls)
    with pytest.raises(ValueError, match="finite numeric vector"):
        env.step(action)
    assert len(stub.calls) == sent


@pytest.mark.parametrize("path", ["/control/step", "/control/reset"])
def test_lost_action_response_blocks_until_explicit_reset(env, stub, path):
    env.reset()
    stub.responders[path] = lambda body: DROP
    call = (lambda: env.step(np.zeros(4))) if path.endswith("step") else env.reset
    with pytest.raises(wec.RecoveryRequiredError, match="uncertain"):
        call()
    sent = stub.paths().count("/control/step")
    with pytest.raises(wec.RecoveryRequiredError, match="uncertain"):
        env.step(np.zeros(4))
    assert stub.paths().count("/control/step") == sent, "an uncertain step must not be re-sent"
    del stub.responders[path]
    env.reset()
    env.step(np.zeros(4))


@pytest.mark.parametrize("call", ["observe", "configure"])
def test_lost_read_only_response_does_not_block_actions(env, stub, call):
    env.reset()
    stub.responders[f"/control/{call}"] = lambda body: DROP
    with pytest.raises(wec.ProtocolError, match="request failed") as error:
        getattr(env, call)()
    assert not isinstance(error.value, wec.RecoveryRequiredError)
    env.step(np.zeros(4))


@pytest.mark.parametrize("reply", [[1, 2], {"obs": OBS}, ("application/json", b"<html>")])
def test_invalid_envelope(env, stub, reply):
    env.reset()
    stub.responders["/control/observe"] = lambda body: reply
    with pytest.raises(wec.ProtocolError):
        env.observe()
    stub.responders["/control/step"] = lambda body: reply
    with pytest.raises(wec.RecoveryRequiredError):
        env.step(np.zeros(4))


def test_oversize_response_is_rejected(env, stub, monkeypatch):
    env.reset()
    monkeypatch.setattr(env_module, "MAX_RESPONSE", 64)
    with pytest.raises(wec.ProtocolError, match="request failed"):
        env.observe()


def step_reply(**change):
    reply = {
        "ok": True,
        "obs": OBS,
        "info": {"episode_id": "e1", "step_id": 1},
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    reply.update(change)
    return reply


@pytest.mark.parametrize(
    "change",
    [
        {"obs": {"robot0_eef_pos": [0, 0], "object_ids": []}},
        {"obs": {"object_ids": []}},
        {"obs": {"robot0_eef_pos": ["x", "y", "z"]}},
        {"obs": {"robot0_eef_pos": [0, 0, 1e39]}},
        {"obs": [1, 2, 3]},
        {"info": None},
        {"terminated": 1},
        {"truncated": "false"},
        {"reward": "high"},
    ],
    ids=[
        "shape",
        "missing-key",
        "strings",
        "float32-overflow",
        "obs-list",
        "info-null",
        "int-terminated",
        "str-truncated",
        "str-reward",
    ],
)
def test_malformed_step_result_requires_recovery(env, stub, change):
    env.reset()
    stub.responders["/control/step"] = lambda body: step_reply(**change)
    with pytest.raises(wec.ProtocolError, match="malformed observation"):
        env.step(np.zeros(4))
    with pytest.raises(wec.RecoveryRequiredError):
        env.step(np.zeros(4))


def test_malformed_observe_does_not_block_actions(env, stub):
    env.reset()
    stub.responders["/control/observe"] = lambda body: {"ok": True, "obs": {}, "info": {}}
    with pytest.raises(wec.ProtocolError, match="malformed observation"):
        env.observe()
    env.step(np.zeros(4))


@pytest.mark.parametrize(
    "code,error",
    [
        ("bad_token", wec.TokenError),
        ("controller_exists", wec.ControllerConflictError),
        ("bad_session", wec.ControllerConflictError),
        ("reset_required", wec.ResetRequiredError),
        ("episode_changed", wec.ResetRequiredError),
        ("teleop_active", wec.TeleopActiveError),
        ("bad_config", wec.BadConfigError),
        ("recording_pending", wec.RecordingPendingError),
        ("instance_mismatch", wec.ProtocolError),
        ("unsupported_protocol", wec.ProtocolError),
        ("not_connected", wec.ProtocolError),
        ("bad_action", wec.WorldEngineClientError),
        ("something_new", wec.WorldEngineClientError),
    ],
)
def test_server_codes_map_to_typed_errors_with_fixes(env, stub, code, error):
    env.reset()
    stub.responders["/control/step"] = lambda body: {
        "ok": False,
        "code": code,
        "error": "nope",
        "fixes": ["do this", "or that"],
    }
    with pytest.raises(error) as caught:
        env.step(np.zeros(4))
    assert caught.value.code == code
    assert "Possible fixes" in str(caught.value) and "2. or that" in str(caught.value)
    # A clean rejection is not an uncertain outcome: the next step is still sent.
    del stub.responders["/control/step"]
    env.step(np.zeros(4))


def test_server_recovery_required_blocks_until_reset(env, stub):
    env.reset()
    stub.responders["/control/step"] = lambda body: {
        "ok": False,
        "code": "recovery_required",
        "error": "render failed",
    }
    with pytest.raises(wec.RecoveryRequiredError, match="render failed"):
        env.step(np.zeros(4))
    del stub.responders["/control/step"]
    with pytest.raises(wec.RecoveryRequiredError, match="uncertain"):
        env.step(np.zeros(4))
    env.reset()
    env.step(np.zeros(4))


def test_rejected_connect_raises_without_binding(stub):
    stub.responders["/control/connect"] = lambda body: {
        "ok": False,
        "code": "bad_token",
        "error": "bad control token",
    }
    with pytest.raises(wec.TokenError):
        wec.connect("wrong", url=stub.url)
    assert stub.paths() == ["/control/connect"], "nothing to release after a failed connect"


def test_connect_rejects_protocol_and_instance_substitution(stub):
    reply = {"ok": True, "session_id": "sess", "proto": 2, "instance_id": "inst", "env": ENV}
    stub.responders["/control/connect"] = lambda body: reply
    with pytest.raises(wec.ProtocolError, match="requested protocol"):
        wec.connect("secret", url=stub.url)
    reply = {**reply, "proto": 3, "instance_id": "other"}
    with pytest.raises(wec.ProtocolError, match="identity mismatch"):
        wec.connect("secret", url=stub.url, expected_instance_id="inst")


def test_configure_applies_now_or_stays_pending(env, stub):
    wide = {**ENV, "action_dim": 7, "active_control_mode": "osc_pose_delta"}
    stub.responders["/control/configure"] = lambda body: {"ok": True, "pending": True, "env": wide}
    assert env.configure(control_mode="osc_pose_delta") == wide
    assert stub.calls[-1][1] == {"control_mode": "osc_pose_delta", "session_id": "sess"}
    assert env.action_dim == 4, "a pending switch must not change the live action space"
    stub.responders["/control/configure"] = lambda body: {"ok": True, "pending": False, "env": wide}
    env.configure(obs_keys=["robot0_eef_pos"])
    assert stub.calls[-1][1]["obs_keys"] == ["robot0_eef_pos"]
    assert env.action_dim == 7 and env.control_mode == "osc_pose_delta"


def test_render_requires_an_image_observation(env):
    with pytest.raises(wec.WorldEngineClientError, match="observe or reset"):
        env.render()
    env.observe()
    with pytest.raises(wec.WorldEngineClientError, match="images=True"):
        env.render()


class Ticks:
    """Stands in for the stop event: ``wait`` returns immediately, n times."""

    def __init__(self, beats):
        self.beats = beats

    def wait(self, timeout):
        assert timeout == 30
        self.beats -= 1
        return self.beats < 0

    def is_set(self):
        return False

    def set(self):
        self.beats = 0


def stop_heartbeat(env):
    env._stop.set()
    env._heartbeat_thread.join(timeout=5)
    assert not env._heartbeat_thread.is_alive()


def test_heartbeat_renews_the_lease_until_stopped(env, stub):
    assert env._heartbeat_thread.is_alive()
    stop_heartbeat(env)
    env._stop = Ticks(3)
    env._heartbeat()
    assert stub.paths().count("/control/heartbeat") == 3
    assert stub.calls[-1][1] == {"session_id": "sess"}
    assert env._heartbeat_error is None


def test_heartbeat_failure_blocks_actions_and_reset_restarts_it(env, stub):
    env.reset()
    stop_heartbeat(env)
    stub.responders["/control/heartbeat"] = lambda body: {
        "ok": False,
        "code": "not_connected",
        "error": "lease expired",
    }
    env._stop = Ticks(5)
    env._heartbeat()
    assert stub.paths().count("/control/heartbeat") == 1, "the loop ends on the first failure"
    assert isinstance(env._heartbeat_error, wec.ProtocolError)
    with pytest.raises(wec.RecoveryRequiredError):
        env.step(np.zeros(4))

    del stub.responders["/control/heartbeat"]
    env._stop = threading.Event()
    env.reset()
    assert env._heartbeat_error is None and env._heartbeat_thread.is_alive()
    env.step(np.zeros(4))


def test_proto2_has_no_heartbeat_or_episode_binding(stub):
    env = wec.connect("secret", url=stub.url, proto=2)
    assert env._heartbeat_thread is None
    env.reset()
    env.step(np.zeros(4))
    assert "expected_episode_id" in stub.calls[-1][1]  # harmless: proto-2 servers ignore it
    env.close()


@pytest.mark.parametrize("call", ["step", "reset", "observe"])
def test_a_connection_the_server_closed_while_idle_is_opened_again(stub, call):
    """Servers close keep-alive connections that stay idle (Node after 5 s).
    The next request went out on the dead socket: a step or reset then raised
    RecoveryRequiredError though it never reached the server. A renderer
    start of more than 5 s did this to the first reset. Now a connection the
    server has closed is opened again before anything is sent."""
    import time

    stub.handler_class.timeout = 0.3  # the server drops connections idle this long
    remote = wec.connect("secret", url=stub.url, timeout=5)
    try:
        remote.reset(seed=0)
        time.sleep(0.8)
        if call == "step":
            remote.step([0, 0, 0, 0])
        elif call == "reset":
            remote.reset(seed=1)
        else:
            remote.observe()
        assert stub.paths()[-1] == f"/control/{call}"
    finally:
        remote.close()


def test_close_survives_a_dead_server(stub):
    env = wec.connect("secret", url=stub.url)
    stub.responders["/control/release"] = lambda body: DROP
    env.close()
    assert env._session_id is None and not env._heartbeat_thread.is_alive()


def test_context_manager_releases(stub):
    with wec.connect("secret", url=stub.url) as env:
        env.observe()
    assert stub.paths()[-1] == "/control/release"
