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
simulates physics and sends the poses with each observation. Nothing is
throttled when no browser tab is open, and the images never travel over the
network. This needs the `render` extra:

```bash
pip install 'world-engine-client[render]'
```

A server with renderer D (render protocol 2) has its scenes drawn by
`WGSLRenderer`: a WGSL splat renderer on wgpu, in this process, with no
browser. It sorts the splats on the GPU, draws every camera in one frame and
reads the images straight into numpy, `(H, W, 3) uint8` per camera, each camera
at its own size. The shaders ship inside this package (`world_engine_client.wgsl`,
a copy of the engine's); the client never runs a shader a server sends. Older
servers (render protocol 1) get the browser path below. `info["render"]` says
which renderer and which adapter drew the step, and `reason` why, when it is
not the obvious one.

Renderer D's pipeline `wgsl/1` draws meshes (objects, robot links) flat for
now: one colour per material, one fixed light, no PBR, environment lighting
or shadows, which a later pipeline version brings. Its mesh pixels differ from
the browser path's (on design doc 16's room scene by 33/255 on average where a
mesh is in front; 1.7/255 in the splat region). `render_options={"renderer":
"chrome"}` keeps the browser path's images. A newer server answers protocol 1
for a scene `wgsl/1` cannot draw (an HDRI, a SuperSplat compressed PLY, a mesh
that is not a GLB) and says why; the client warns once with that reason.

Which adapter draws: a GPU first (a discrete one, an integrated one, then a
virtual GPU of a VM or a cloud), then a software one. On Linux the client
tries the system's Vulkan loader with the GPU's driver, the bundled loader with
the system's drivers (a container that has the NVIDIA driver but no
libvulkan1), Mesa's lavapipe (`apt install mesa-vulkan-drivers`), and last the
bundled SwiftShader. The bundled loader and SwiftShader come in the package
`world-engine-swiftshader`, which the extra installs on Linux x86_64 and
aarch64. On Windows the order is the GPU, then WARP (Windows' software
renderer); on macOS, Metal. Each step is tried in a short-lived subprocess,
and the result is kept for a day next to the render cache, so a later connect
does not try again. With a software adapter, one `RuntimeWarning` says so;
expect about a step per second on a real scene. A second warning comes after a
warm-up when the rendering is slower than real time (29.4 steps per second, one
step per 0.034 s), naming the adapter, its type and the measured rate. On a
machine with an NVIDIA GPU that Vulkan cannot reach, the warning says to
install `libx11-6 libxext6 libegl1` and, in a container, to add `graphics` to
`NVIDIA_DRIVER_CAPABILITIES`. The probes import wgpu from where your process
does (its `sys.path`), so `pip install --user` and `PYTHONPATH` installs work.

A Linux machine can show a GPU (a `/dev/dri/renderD*` node, `/dev/nvidiactl`)
that Vulkan does not reach: a GPU without its Vulkan driver
(`apt install mesa-vulkan-drivers` brings the Intel and AMD ones), a virtual
GPU, an old one. Renderer D would then draw on the CPU, while Chrome may reach
the GPU through OpenGL, an order of magnitude faster. So in that case the
client first starts the browser path with hardware WebGL only: if Chrome
reaches the GPU, it draws, with one `RuntimeWarning` that says how to get
renderer D onto the GPU; if not, renderer D draws on the CPU, and the client
remembers that for a day. `render_options={"renderer": "wgsl"}`, or an
`adapter` of `"software"` or `"name:..."`, keeps renderer D.

When no adapter renders, or renderer D cannot draw the scene (a PLY or GLB it
does not read, more splats than the GPU holds), the browser path is tried
(headless Chrome with its own SwiftShader) with a `RuntimeWarning`, and when
that fails too, connect raises `RendererError` with the fixes:
`apt install libvulkan1 mesa-vulkan-drivers`, the container settings, or
`images=False` for a state-only session.

`render_options` for renderer D: `adapter` picks the adapter (`"auto"`,
`"low-power"` for an integrated GPU first, `"software"`, or `"name:Intel"` for
an adapter whose name contains "Intel"; the `WE_RENDER_ADAPTER` environment
variable sets the default), `allow_software=False` refuses software adapters,
`color_format` overrides the colour target. `renderer` (`"auto"`, `"wgsl"` or
`"chrome"`; default `WE_RENDERER` or `"auto"`) forces one path: `"wgsl"` never
falls back to Chrome, `"chrome"` offers the server only the browser path.
`env.last_timings` splits each step into server, transport and render time.

The extra includes `cryptography`, which reads the platform's encrypted scene
assets. A session with a local `world-engine serve` works without it.

### The browser path

For a server from before renderer D, for a scene renderer D's pipeline does
not draw, and as the last fallback, a headless Chrome here runs the engine's
capture page, the same scene the browser viewer shows. An installed Google
Chrome is used if there is one. Otherwise the browser path uses Playwright's
headless shell: Chrome Headless Shell 148, the build of playwright 1.60.0,
which the `render` extra pins. The client downloads it once (about 120 MB on
Linux and Windows, 100 MB on macOS; 260 MB on disk on Linux) into a directory
of its own next to the render cache, `<user cache dir>/digiarena/browsers`
(`DIGIARENA_BROWSERS_DIR` moves it). Playwright's own browser directory is
neither used nor changed. The download starts in the background when the
renderer starts, while the scene downloads, and prints nothing: it is logged
to the `world_engine_client.render` logger. Set `WE_RENDER_AUTO_INSTALL=0` or
pass `render_options={"auto_install": False}` to turn it off, and run
`python -m world_engine_client.render install` to download it yourself
(`python -m world_engine_client.render where` shows what is there).
`WE_RENDER_BROWSER=chrome` or `WE_RENDER_BROWSER=shell` uses only that one.
On a minimal Linux system the shell may also need Chromium's system
libraries: `python -m playwright install-deps chromium` (needs sudo).
`info["render"]` names the browser and its version.

The `WE_RENDER_CHROME_ARGS` environment variable overrides Chrome's flags.

Chrome reads the page and the scene from a loopback HTTP server in this
process and sends its images back over a loopback WebSocket. Other processes
on the machine can reach both ports, so both need a random token of this
session: Playwright adds it to every request the page makes, and the
WebSocket URL the page gets ends in it. Requests and connections without it
get 403. Without the token, another local process cannot read the scene or
put its own images into your observations. For the page, the loopback server
fetches only the page and the files of its build from the server.

### The render cache

The client downloads the scene's assets (and, for the browser path, the
capture page's scripts) before anything is drawn and keeps them in the render
cache, so the next session of the same scene downloads nothing. Renderer D
decrypts each asset into memory and uploads it to the GPU; nothing plain is
written to disk.
The cache is `<user cache dir>/digiarena/render/`: `$XDG_CACHE_HOME` or
`~/.cache` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows.
`DIGIARENA_CACHE_DIR` or `render_options={"cache_dir": "..."}` moves it.
Its directories are 0700 and its files 0600. `close()` no longer deletes
anything.

The DigiArena platform sends scene assets encrypted (DAC1: AES-256-GCM under
a key for your account and that scene). The client stores them as they
arrive and asks the platform for the key at every connect; the key stays in
memory. So the scene's files are never in the clear on disk, and its cache
stops working the moment you lose access: the scene made private, your key
revoked, or a new key epoch on the platform. The robot meshes (public MuJoCo
Menagerie geometry) and the capture page's scripts are kept plain. A local
`world-engine serve` sends your own scene as it is, and the cache keeps it
plain.

What the encryption does: it keeps the scene from someone who can read your
disk but cannot sign in as you. It does not keep it from you. Your key
decrypts it, and while a session runs the decrypted assets and the images
are in the memory of this process (and of Chrome, on the browser path) and of
the GPU.

Each file is checked when a session uses it, and a damaged or changed file
is downloaded again. Processes that share the cache download a file once. A
process that dies during a download does not hold up the others. A broken
download resumes where it stopped; one that sends nothing for 60 seconds
fails. Downloads do not count against the renderer's start time.

A file not used for 30 days is deleted unless its scene is pinned. The check
runs in the background when a renderer starts, at most once a day, after the
session's own files are in place: the scene you open is not deleted, however
long it went unused. A machine that no longer starts renderers keeps its files
until you clear them.

```python
from world_engine_client import cache

cache.list()                     # [{scene, scope, size, last_used, pinned, ...}]
cache.pin("alice/kitchen_001")   # keep a scene however long it goes unused
cache.unpin("alice/kitchen_001")
cache.clear("alice/kitchen_001") # or clear() for everything
cache.configure(max_age_days=30, max_size_gb=None)  # 0 days: keep everything
```

From a shell: `python -m world_engine_client.cache ls`, `pin <scene>`,
`unpin <scene>`, `clear <scene>` or `clear --all`, and `config
--max-age-days N --max-size-gb N`. `--dir PATH` picks another cache.
`DIGIARENA_CACHE_MAX_AGE_DAYS` and `DIGIARENA_CACHE_MAX_SIZE_GB` override the
stored settings. The DigiArena SDK has the same as `da.cache` and
`digiarena cache`.

Clients before 0.6 kept plain copies of scene assets in `cache_dir` and in
`we_render_*` temporary directories. This client does not read them. It
deletes `we_render_*` directories that nobody has used for a day, and
`python -m world_engine_client.cache --dir <old cache_dir> clear --legacy`
deletes the rest, except directories used in the last hour: an old client may
still be loading from one. `clear --all --legacy` clears the cache as well.

On the browser path, without a hardware GPU, the images are rendered in
software (SwiftShader) and a `RuntimeWarning` says so before the first frame. Expect seconds per step: with
four CPU cores, a 400k-splat scene took 6–7 s per step and a 1.6M-splat scene
25–28 s. `render_options={"allow_software": False}` turns software rendering
off, and then a machine without hardware WebGL raises `RendererError`. If this
machine has a GPU and still renders in software, check `chrome://gpu`.

Starting the renderer may take `timeout` + 30 seconds (330 s by default) to
launch Chrome and load the scene, over all of Chrome's flag sets together, and
`timeout` more for a software first frame. Downloading the browser does not count.
`render_options={"timeout": 600}` gives a slow machine more time.

On the browser path on a hardware adapter, the client times the rendering of
the first steps (10 steps, after 2 untimed ones). If the median is below 3
steps per second, it
issues one `RuntimeWarning` per process with the adapter's name and the measured
rate, and suggests `images=False` or fewer cameras. The threshold comes from
measurement: on an RTX 4090 with three 256×256 cameras, this renderer drew
0.4M–1.6M splat scenes at 6.8–17.2 steps per second (render only, measured
2026-10-26 on a GPU that also ran a desktop), and 3 is about half of the slowest
of these. Network latency does not count, because only the rendering is timed.
Real time (29.4 steps per second, one step per 0.034 s) is not the threshold
there: the browser path stays below it even on an RTX 4090. Renderer D draws the
same scenes at about 150 steps per second there.

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

`step()` returns the task's reward, and `env.metadata.get("reward_type")`
names its kind: `"sparse"`, `"dense"` or `"staged"`. It is `None` without a
task, and servers from before staged rewards do not send it. A `sparse` reward
pays `success_reward` on success and a `dense` reward is the task template's
distance shaping. A `staged` reward pays stage points. The server divides it
by the task's total points, so an episode that finishes sums to 1.0. This is
the default, and older clients get the same.

To get a staged reward in points, connect with `normalize_reward=False`:

```python
env = we_client.connect(token, url=..., normalize_reward=False)
```

Each reward is then multiplied by `info["task_progress"]["reward"]["max"]`, the
task's total points. A step pays the points it earned, less the failure penalty
on the step that fails. With partial credit on (the task's
`reward.partial_credit`), a stage pays part of its points while some of its
conditions hold. A step on which one stops holding gives that part back: its
reward is negative, though nothing failed and no penalty is taken. A step never
pays the running total again, so holding a state that already scored earns
nothing more. The running totals, in points, are in
`info["task_progress"]["reward"]`: `earned`, `max`, `penalty` and
`earned_by_stage`. After every step of an episode that started with `reset()`,
the rewards so far add up to `earned - penalty` of that step's info, up to
floating-point rounding (compare with `math.isclose`).

- The points of stages that end at reset come with the first step.
  `info["task_progress"]["completed_at"]` gives those stages step 0.
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
`WE_TEST_REQUIRE_CHROME=1` makes it fail instead. The tests use the headless
shell in your `<user cache dir>/digiarena/browsers` (or `DIGIARENA_BROWSERS_DIR`)
and never download one; `WE_RENDER_BROWSER=shell` makes them use it rather than
an installed Google Chrome.

`bench/igpu/` (see its README) times renderer D against the browser path on
this machine's GPU and compares their images with an RTX 4090's: the bench for
integrated GPUs, which neither CI nor the development machine has.

`tests/test_render_wgsl.py` draws a small scene with renderer D through
`connect()`, on whatever adapter the machine has; it is skipped without one,
and `WE_TEST_REQUIRE_WGSL=1` makes it fail instead.
`WE_TEST_EXPECT_ADAPTER=<step>` (`system`, `bundled-loader`, `lavapipe`,
`swiftshader`, `warp`) also checks which step of the adapter search drew. CI
runs it on lavapipe and on the bundled SwiftShader alone (`test.yml`), on
Windows' WARP and on Linux arm64 (`software-render.yml`); `gpu.yml` is the GPU
tier, run by hand on a self-hosted runner.

Renderer D's shaders and host core come from the engine repository
(`frontend/src/render/wgsl/`, `world_engine/renderer/wgsl/`): change them there,
then copy them here with `python scripts/sync_wgsl.py <engine checkout>`.
`wgsl/SOURCE.json` records the engine commit and every file's SHA-256;
`tests/test_wgsl_vendored.py` checks the copy against it, and `--check`
compares the copy with an engine checkout before a release.

`world-engine-swiftshader` lives in `swiftshader/` with its own `pyproject.toml`
and its own release workflow (`publish-swiftshader.yml`, tags
`swiftshader-v<version>`). Publish it before a world-engine-client release that
needs a new version of it; `publish.yml` stops a release whose render extra
names a version that is not on PyPI yet. `python scripts/render_size.py
--wheel-dir <wheels>` reports the render extra's download size.
