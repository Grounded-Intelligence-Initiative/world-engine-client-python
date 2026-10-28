# world-engine-client

Remote Gym-style Python client for a World Engine
simulation server — a Gaussian-Splatting real-to-sim robot simulation framework.
Your Python connects with a token, becomes the (single) controller, and drives
the MuJoCo physics with `reset`/`step`, while every browser watching the
server's `viewer.html` sees the episode live.

```bash
pip install world-engine-client
```

Needs Python 3.11 or newer. Base dependencies: numpy only. Install
`world-engine-client[img,gym]` for optional Pillow decoding and the Gymnasium adapter. The import
name is `world_engine_client` (the `world_engine` module name belongs to the simulation server
package).

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
env = we_client.connect("<token from the Inspector>", url="http://<host>:8080",
                        mode="resume", images=True)
obs, info = env.observe()  # preserves the current attempt; no physics advance
obs, reward, term, trunc, info = env.step([0.2, 0, 0, 0, 0, 0, 0])
```

Resume attachment rejects active human teleoperation with `TeleopActiveError`.
For hosted DigiArena sessions use `digiarena.attach(hosted_session_id)`; browser
cookies and editor write credentials are not Python control credentials.

Protocol 3 is the default. Direct connections remain reset-required and state-only
unless requested otherwise. `proto=2` preserves state-only compatibility and
rejects image/resume requests. `images=True` captures all configured cameras;
`cameras=['wrist']` selects a subset. RGB arrays appear in `obs['rgb']`; `render()`
returns the latest camera without stepping. `as_gymnasium()` supplies an optional
Gymnasium Env.

### Camera images

Camera images are drawn on your computer, never on the server: the server only
simulates physics and sends the poses with each observation, and a headless Chrome
here runs the same scene the browser viewer shows. Nothing is throttled when no
browser tab is open, and the images never travel over the network. This needs the
`render` extra:

```bash
pip install 'world-engine-client[render]'
```

An installed Google Chrome is used if there is one. Otherwise the first session
downloads Playwright's Chromium (about 300 MB, once, with progress in the
terminal). Set `WE_RENDER_AUTO_INSTALL=0` or pass
`render_options={"auto_install": False}` to turn that off and run
`python -m playwright install chromium` yourself. On a minimal Linux system
Chromium may also need its system libraries:
`python -m playwright install --with-deps chromium` (needs sudo).

Scene assets are fetched once per session and cached by content hash in a
temporary directory that `close()` deletes; pass
`render_options={"cache_dir": "..."}` to keep them across sessions. The
`WE_RENDER_CHROME_ARGS` environment variable overrides Chrome's flags.
`env.last_timings` splits each step into server, transport and render time.

Without a hardware GPU, the images are rendered in software (SwiftShader) and
a `RuntimeWarning` says so before the first frame. Expect seconds per step: with
four CPU cores, a 400k-splat scene took 6–7 s per step and a 1.6M-splat scene
25–28 s. `render_options={"allow_software": False}` turns software rendering
off, and then a machine without hardware WebGL raises `RendererError`. If this
machine has a GPU and still renders in software, check `chrome://gpu`.

Starting the renderer may take `timeout` + 30 seconds (330 s by default) to
launch Chrome and load the scene, over all of Chrome's flag sets together, and
`timeout` more for a software first frame. Downloading Chromium does not count.
`render_options={"timeout": 600}` gives a slow machine more time.

On a hardware adapter, the client times the rendering of the first steps
(10 steps, after 2 untimed ones). If the median is below 3 steps per second, it
issues one `RuntimeWarning` per process with the adapter's name and the measured
rate, and suggests `images=False` or fewer cameras. The threshold comes from
measurement: on an RTX 4090 with three 256×256 cameras, this renderer drew
0.4M–1.6M splat scenes at 6.8–17.2 steps per second (render only, measured
2026-10-26 on a GPU that also ran a desktop), and 3 is about half of the slowest
of these. Network latency does not count, because only the rendering is timed.
Real time (29.4 steps per second, one step per 0.034 s) is not the threshold:
this renderer stays below it even on an RTX 4090.

Heartbeats run every 30 seconds with a 90-second controller lease. They start at
connect, before the renderer, so a renderer start longer than the lease keeps
the session. Expiry freezes the simulation until explicit takeover. A lost
action response is never retried:
`RecoveryRequiredError` blocks further steps until explicit reset/reconnection.
Pending recording data blocks reset with `RecordingPendingError` and is preserved.

## Action & observation space

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
- `env.action_space` / `env.observation_space` are Gymnasium spaces for the
  negotiated control mode and cameras (needs the `gym` extra); `env.as_gymnasium()`
  returns a full `gymnasium.Env`.
- `RemoteEnv` is a context manager: `with we_client.connect(...) as env:`
  releases the controller lock on exit.
- Errors are typed (`TokenError`, `ControllerConflictError`,
  `ResetRequiredError`, `TeleopActiveError`, `BadConfigError`) and carry the
  server's "Possible fixes" hints.

## Rewards

`step()` returns the task's reward, and `env.metadata["reward_type"]` names
its kind. A `sparse` reward pays `success_reward` on success and a `dense`
reward is the task template's distance shaping. A `staged` reward pays stage
points. The server divides it by the task's total points, so an episode that
finishes sums to 1.0. This is the default, and older clients get the same.

To get a staged reward in points, connect with `normalize_reward=False`:

```python
env = we_client.connect(token, url=..., normalize_reward=False)
```

Each reward is then multiplied by `info["task_progress"]["reward"]["max"]`, the
task's total points. A step pays the points it earned, less the failure penalty
on the step that fails. It never pays the running total again, so holding a
state that already scored earns nothing more. The running totals, in points,
are in `info["task_progress"]["reward"]`: `earned`, `max`, `penalty` and
`earned_by_stage`. After every step of an episode that started with `reset()`,
the rewards so far add up to `earned - penalty` of that step's info, up to
floating-point rounding (compare with `math.isclose`).

- The points of stages that already hold at reset come with the first step.
- An episode decided at reset still takes one step. It does not move the world,
  returns `terminated=True` and the reset's info, and pays those points.
- After attaching with `mode="resume"`, the rewards cover the steps this client
  took. The points paid before it attached are not in the sum.
- `as_gymnasium()` and wrappers such as `RecordEpisodeStatistics` see the same
  rewards, so `info["episode"]["r"]` is in points too.
- `sparse` and `dense` rewards are the same with either setting.
- The switch is unrelated to Gymnasium's `NormalizeReward` wrapper, which
  scales rewards by a running estimate of the return's spread. The wrapper
  works on top of either setting.

With `normalize_reward=False`, a staged step whose info has no usable
`task_progress.reward.max` raises `ProtocolError`, and the next step needs a
`reset()`. A server with staged rewards always sends it.

## Development

```bash
pip install -e '.[img,gym,render]' pytest && python -m pytest
```

`tests/` covers the client alone against a stub server: binary decoding, the
never-retry / recovery rules, heartbeats and Gymnasium spaces. The wire-contract
tests (real client against a real server) live in the server repo:
`tests/test_remote_*.py` and `tests/test_teleop_channel.py`.
`tests/test_render_chrome.py` renders a small page in a real headless Chrome on
SwiftShader. It is skipped without the `render` extra or a browser;
`WE_TEST_REQUIRE_CHROME=1` makes it fail instead.
