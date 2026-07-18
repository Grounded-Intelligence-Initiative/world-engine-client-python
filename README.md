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
    action = my_policy(obs)   # [-1,1]^7: [dpos(3), drot_axisangle(3), gripper]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

### Connecting to an editor Simulate session

While the scene editor is running a **Simulate** preview, the same `/control`
API is live against the exact scene on the editor's screen. The control token
is shown in the editor Inspector's **Python API** card (copy / edit /
regenerate it there):

```python
env = we_client.connect("<token from the Inspector>", url="http://<host>:8080")
obs, info = env.reset()
obs, reward, term, trunc, info = env.step([0.2, 0, 0, 0, 0, 0, 0])
```

If the editor user has **teleop enabled**, `step()` raises `TeleopActiveError`
(the human owns the environment) — ask them to disable teleop in the
Inspector, or wait. `connect()`/`reset()` remain allowed.

## Action & observation space (proto 2)

The default control mode is **`osc_pose_delta`** — robosuite OSC_POSE
conventions:

- action = `[dpos(3), drot_axisangle(3), gripper(1)]`, all in `[-1, 1]`;
- `dpos`: EE translation delta, ±1 → ±`ee_step` m/step (default 0.05);
- `drot_axisangle`: world-frame axis-angle rotation delta, ±1 → ±`ee_rot_step`
  rad/step (default 0.5);
- `gripper`: a sign-driven rate — each step moves the internal command by 0.2
  in the sign's direction (full open↔close ≈ 10 steps); 0 = no change.

Observations follow robosuite naming: `robot0_eef_pos` (3), `robot0_eef_quat`
(4, scalar-last **xyzw**), `robot0_gripper_qpos`, `robot0_joint_pos`,
`robot0_joint_vel`, `robot0_sew_pos` (9, shoulder/elbow/wrist world keypoints,
when a preset robot is attached), `object` (pos(3)+quat(4, xyzw) per body,
sorted-id order). The proto-1 keys (`proprio` with wxyz quat, `object_state`,
`joint_positions`) remain as compatibility aliases for one release.

### Negotiating the space

```python
# at connect:
env = we_client.connect(token, url=..., control_mode="osc_pose_abs",
                        obs_keys=["robot0_eef_pos", "robot0_eef_quat"])

# or mid-run (lands at the next reset if an episode is active):
env.configure(control_mode="osc_pose_delta")
obs, info = env.reset()          # the switch is live here
print(env.action_dim, env.control_mode, env.observation_keys)
```

Supported modes are advertised in `env.metadata["supported_control_modes"]`
with per-mode schemas in `env.metadata["control_mode_schemas"]`:

- `osc_pose_delta` — 7-dim normalized deltas (default);
- `osc_pose_abs` — 7-dim absolute `[pos(3), rot_axisangle(3), gripper]`
  targets (unscaled, robosuite v1.5 `input_type="absolute"`);
- `ee_pos_delta` — the legacy 4-dim `[dx, dy, dz, gripper]` action.

Unknown names raise `BadConfigError` listing the supported values.

### 6D rotation helpers

Demonstration HDF5s recorded by the server carry 6D rotations
(`action_dict/*_rot_6d`, robomimic convention: first two rotation-matrix rows,
Gram-Schmidt decode). Pure-numpy helpers ship here:

```python
from world_engine_client import rot6d_to_matrix, matrix_to_rot6d
R = rot6d_to_matrix(demo["action_dict/abs_rot_6d"][t])
```

## Semantics

- Exactly one controller may be bound; `connect(token, takeover=True)` revokes
  a dead session.
- After `terminated`/`truncated` you must `reset()` (Gym semantics — the server
  refuses further steps with `reset_required`).
- `RemoteEnv` is a context manager: `with we_client.connect(...) as env:`
  releases the controller lock on exit.
- Errors are typed (`TokenError`, `ControllerConflictError`,
  `ResetRequiredError`, `TeleopActiveError`, `BadConfigError`) and carry the
  server's "Possible fixes" hints.

## Development

```bash
pip install -e . && python -m pytest
```

The wire-contract tests (real client against a real server) live in the server
repo: `tests/test_remote_control.py` and `tests/test_teleop_channel.py`.
