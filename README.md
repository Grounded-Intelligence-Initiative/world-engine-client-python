# world-engine-client

Remote Gym-style Python client for a World Engine
simulation server — a Gaussian-Splatting real-to-sim robot simulation framework.
Your Python connects with a token, becomes the (single) controller, and drives
the MuJoCo physics with `reset`/`step`, while every browser watching the
server's `viewer.html` sees the episode live.

```bash
pip install world-engine-client
```

Dependencies: numpy only. The import name is `world_engine_client` (the
`world_engine` module name belongs to the simulation server package).

## Usage

Server side (prints the control token):

```bash
world-engine --scenes-dir scenes serve --mode simulator \
    --scene <scene> --task <task> --allow-remote-control
```

Client side:

```python
import world_engine_client as we_client

env = we_client.connect("<control token>", url="http://<host>:8080")
obs, info = env.reset(seed=0)
for _ in range(200):
    action = my_policy(obs)                      # [-1,1]^4: [dx, dy, dz, gripper]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

`obs` is a dict of numpy arrays: `proprio` (8: ee pos + quat wxyz + gripper),
`joint_positions`, `object_state` (pos+quat per body, sorted-id order) plus
`object_ids`. Camera-image observations ride the server's headless-renderer
path and are not yet exposed here (see the server repo's
`docs/remote_inference_api.md` for the design).

## Semantics

- Exactly one controller may be bound; `connect(token, takeover=True)` revokes
  a dead session.
- After `terminated`/`truncated` you must `reset()` (Gym semantics — the server
  refuses further steps with `reset_required`).
- `RemoteEnv` is a context manager: `with we_client.connect(...) as env:`
  releases the controller lock on exit.
- Errors are typed (`TokenError`, `ControllerConflictError`,
  `ResetRequiredError`) and carry the server's "Possible fixes" hints.

## Development

```bash
pip install -e . && python -m pytest
```

The wire-contract tests (real client against a real server) live in the server
repo: `tests/test_remote_control.py`.
