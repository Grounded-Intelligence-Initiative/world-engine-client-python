"""Client rendering without a browser: the extra requirement, the render_state
hand-off, credentials on asset requests and the digest cache."""

import copy
import hashlib

import numpy as np
import pytest
from conftest import ENV, OBS

import world_engine_client as wec
from world_engine_client import render as render_module

CAMERAS = {"front": {"width": 8, "height": 6}}
STATE = {"type": "state", "frame_id": 0, "bodies": [], "robot": {}, "cameras": []}
BUNDLE = {
    "protocol": 1,
    "scene_init": {"type": "scene_init", "cameras": []},
    "digests": {"/__asset/0.ply": "ab" * 32},
}


class FakeRenderer:
    instances = []

    def __init__(self, fetch, scene_init, digests, **options):
        self.fetch, self.scene_init, self.digests = fetch, scene_init, digests
        self.options = options
        self.states, self.closed, self.fail = [], False, False
        FakeRenderer.instances.append(self)

    def render(self, state):
        if self.fail:
            raise render_module.RendererError("GPU lost", code="renderer_unavailable")
        self.states.append(state)
        return {"front": np.full((6, 8, 3), len(self.states), np.uint8)}

    def close(self):
        self.closed = True


@pytest.fixture
def client_render(stub, monkeypatch):
    """A stub that offers client rendering; LocalRenderer replaced by a fake."""
    FakeRenderer.instances = []
    monkeypatch.setattr(render_module, "available", lambda: True)
    monkeypatch.setattr(render_module, "LocalRenderer", FakeRenderer)
    env = {**copy.deepcopy(ENV), "cameras": CAMERAS, "render": "client"}
    stub.responders["/control/connect"] = lambda body: {
        "ok": True,
        "session_id": "sess",
        "proto": 3,
        "instance_id": "inst",
        "episode_id": "e0",
        "env": env,
        **({"render": BUNDLE} if body.get("render") == "client" else {}),
    }
    for path in ("/control/reset", "/control/observe"):
        stub.responders[path] = lambda body: {
            "ok": True,
            "obs": dict(OBS),
            "info": {"episode_id": "e1", "step_id": 0},
            "render_state": STATE,
        }
    stub.responders["/control/step"] = lambda body: {
        "ok": True,
        "obs": dict(OBS),
        "info": {"episode_id": "e1", "step_id": 1},
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "render_state": STATE,
    }
    return stub


def test_images_are_rendered_locally(client_render):
    env = wec.connect("secret", url=client_render.url, images=True, render_options={"x": 1})
    connect = client_render.calls[0][1]
    assert connect["render"] == "client" and connect["images"] is True
    fake = FakeRenderer.instances[0]
    assert fake.scene_init == BUNDLE["scene_init"] and fake.options == {"x": 1}
    obs, _ = env.reset(seed=0)
    assert obs["rgb"]["front"].shape == (6, 8, 3) and fake.states == [STATE]
    assert env.render() is obs["rgb"]["front"]
    env.close()
    assert fake.closed and client_render.paths()[-1] == "/control/release"


@pytest.mark.parametrize("kw", [{"images": True}, {"cameras": ["front"]}])
def test_images_without_the_extra_fail_before_any_request(stub, monkeypatch, kw):
    monkeypatch.setattr(render_module, "available", lambda: False)
    with pytest.raises(wec.RendererError, match=r"render\]"):
        wec.connect("secret", url=stub.url, **kw)
    assert stub.calls == []


@pytest.mark.parametrize("mode", ["server", "browser"])
def test_servers_never_render(stub, mode):
    with pytest.raises(ValueError, match="no longer render"):
        wec.connect("secret", url=stub.url, images=True, render=mode)
    assert stub.calls == []


def test_a_server_without_client_rendering_is_released(client_render):
    client_render.responders["/control/connect"] = lambda body: {
        "ok": True,
        "session_id": "sess",
        "proto": 3,
        "instance_id": "inst",
        "episode_id": "e0",
        "env": {**copy.deepcopy(ENV), "cameras": CAMERAS},
    }
    with pytest.raises(wec.ProtocolError, match="predates client rendering"):
        wec.connect("secret", url=client_render.url, images=True)
    assert client_render.paths()[-1] == "/control/release"


def test_a_render_failure_leaves_physics_and_observe_renders_again(client_render):
    env = wec.connect("secret", url=client_render.url, images=True)
    env.reset(seed=0)
    fake = FakeRenderer.instances[0]
    fake.fail = True
    with pytest.raises(wec.RendererError, match="GPU lost"):
        env.observe()
    fake.fail = False
    obs, _ = env.observe()
    assert obs["rgb"]["front"][0, 0, 0] == 2
    obs, *_ = env.step([0, 0, 0, 0])  # the failed render blocked no action
    assert obs["rgb"]["front"][0, 0, 0] == 3
    env.close()


def test_an_observation_without_poses_is_a_protocol_error(client_render):
    env = wec.connect("secret", url=client_render.url, images=True)
    client_render.responders["/control/observe"] = lambda body: {
        "ok": True,
        "obs": dict(OBS),
        "info": {"episode_id": "e1", "step_id": 0},
    }
    with pytest.raises(wec.ProtocolError, match="malformed observation"):
        env.observe()
    env.close()


def test_asset_requests_carry_the_session_credentials(client_render):
    env = wec.connect(
        "secret", url=client_render.url, images=True, access_provider=lambda: "scoped"
    )
    client_render.files["/render/index.html"] = b"<html>"
    with env._open_render("/index.html") as response:
        assert response.read() == b"<html>"
    path, _, headers = client_render.calls[-1]
    assert path == "/render/index.html"
    assert headers["X-We-Control-Token"] == "secret"
    assert headers["Authorization"] == "Bearer scoped"
    with pytest.raises(FileNotFoundError):
        env._open_render("/missing.js")
    env.close()


def test_downloads_are_verified_against_their_digest(tmp_path, stub):
    renderer = object.__new__(render_module.LocalRenderer)  # no browser needed
    env = wec.connect("secret", url=stub.url)
    renderer._fetch = env._open_render
    data = b"splat bytes"
    stub.files["/render/__asset/0.ply"] = data
    good = tmp_path / hashlib.sha256(data).hexdigest()
    renderer._download("/__asset/0.ply", good.name, good)
    assert good.read_bytes() == data
    bad = tmp_path / ("0" * 64)
    with pytest.raises(wec.RendererError, match="digest"):
        renderer._download("/__asset/0.ply", bad.name, bad)
    assert not bad.exists() and not bad.with_suffix(".part").exists()
    env.close()
