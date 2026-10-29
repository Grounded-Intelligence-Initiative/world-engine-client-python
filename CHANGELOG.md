# Changelog

## 0.7.1

Scene assets straight from the platform's storage, three fixes, and
documentation.

- When the DigiArena platform offers it, scene assets come straight from its
  storage. A request for one says that the client follows a redirect
  (`X-We-Render-Redirect: 1`), and the platform may answer with a URL there,
  valid for minutes. The client fetches it on a new connection, and of the
  request's headers only the Range goes along, so the session's token and
  Authorization header stay with the platform; an https platform's URL must be
  https too. The bytes are the same DAC1 file, checked and cached as before. A
  URL that fails is not used again; the platform signs a new one, and the
  download resumes where it stopped. Where storage cannot be reached (10 s to
  connect), or after its second failure, the session streams its scene assets
  from the platform, as 0.7.0 does, and the `world_engine_client.env` logger
  says so in one INFO line. Clients up to 0.7.0 send no such header and keep
  getting the stream.
- Renderer D frees its GPU device when the session closes. In wgpu-py 0.32 a
  device and its queue refer to each other, so up to 0.7.0 the device, its
  queue and its adapter outlived `close()` until Python's cyclic garbage
  collector ran: 50 sessions in one process, without a collection, grew it
  from 222 to 1,201 MiB, from 65 to 113 threads and from 34 to 154 file
  descriptors. `close()` now clears those references, so all three go when
  it returns. If the device is still there after that (another wgpu-py), it
  runs one collection, except while the interpreter shuts down. Calling it
  again does nothing. On an RTX 4090, 50 sessions of the tests' scene stay
  at 230 to 239 MiB, 66 threads and 35 file descriptors; 20 sessions grew
  0.7.0 from 227 to 1,831 MiB, 66 to 146 threads and 35 to 235 file
  descriptors. The device of an adapter renderer D failed on, and of a scene
  it could not draw, also goes as soon as the next adapter or the browser
  path takes over, wgpu's errors included: wgpu-py raises them in a
  reference cycle with its own frame, which kept the device too. After a
  render that wgpu failed, `close()` frees the device even while the error
  is still being handled, as at the end of a `with` block.
- A client without wgpu, upgraded without the render extra
  (`pip install -U digiarena`, or `pip install -U world-engine-client`
  alone), still offers the server only the browser path, and now says so:
  one `RuntimeWarning` per process that renderer D needs the render extra,
  with the command, `pip install -U 'world-engine-client[render]'`, and the
  reason in `info["render"]["reason"]` of every session. Neither appears when
  the caller asks for the browser path (`render_options={"renderer":
  "chrome"}` or `WE_RENDERER=chrome`).
- With `adapter="software"`, or a software adapter by name, the software
  warning says that renderer D draws on the CPU as asked, and how fast. It
  used to say that Vulkan reaches no GPU, and on a machine whose NVIDIA GPU
  Vulkan does reach it said to install the libraries the NVIDIA Vulkan driver
  loads. The warning for rendering slower than real time gave the same NVIDIA
  advice; for these callers it now leaves it out. `info["render"]["reason"]`
  says the adapter was asked for, not "no hardware adapter".

The README and the docstrings now cover what an engine with mobile bases and
several robots sends, which 0.7.0 already handles. The numbers below were
checked with 0.7.0 against such an engine, in a TidyBot scene, a scene with
two Pandas and scenes with a TidyBot and a Panda in either order.

- Scenes with a mobile base: a connect without `control_mode` gets
  `osc_pose_delta_base_vel`, `Box(-1, 1, (10,))`, laid out
  `[dpos(3), drot(3), gripper(1), base_vx, base_vy, base_wz]`. The arm's part
  is `osc_pose_delta`'s in the robot's base frame (±1 is 0.05 m and 0.5 rad
  per step), the gripper is dim 6, and ±1 on the base is 0.5 m/s forward or
  sideways and 3.14 rad/s of yaw; the base accelerates at most 0.5 m/s² and
  2.36 rad/s². A 7-value policy passes `control_mode="osc_pose_delta"`;
  without it, `step()` raises `ValueError` before sending anything. The
  README lists the five `robot{i}_base_*` observations.
- Scenes with several robots: the action is every robot's in scene order
  (14 values for two Pandas, 17 for a TidyBot with a Panda), the schemas carry
  `robots`, prefixed `parts`, `gripper_index` and `gripper_indices`, and the
  observations add `robot{i}_*` for each robot after the first.
- Cameras: `available_cameras` and `cameras` in the metadata, the default
  `wrist` (128×128), `robot{i}_wrist` for each robot after the first, and a
  TidyBot's `robot{i}_wrist` (256×192) and `robot{i}_base` (256×144),
  `robot0_*` when it is the first robot, which follow the links they ride. A
  scene camera wins over a default camera of the same name. Renderer D draws
  each at its own size.
- The README says to upgrade with the extra,
  `pip install -U 'world-engine-client[render]'`: pip does not remember
  extras, and a client without wgpu offers the server only the browser path
  (the second fix above).
- `robot0_eef_target_pos` and `robot0_eef_target_quat`, the commanded pose,
  are in the observation list; `robot0_eef_pos` and `robot0_eef_quat` are the
  measured one.
- The docstrings of the package, `connect()`, `RemoteEnv.step()` and
  `RemoteEnv.action_space` say the same.

## 0.7.0

- Renderer D: camera images from a server with renderer D (render protocol 2)
  are drawn by `world_engine_client.render_wgsl.WGSLRenderer`, a WGSL renderer
  on wgpu, in this process and without a browser. Splats are sorted on the
  GPU, every camera is drawn in one frame, and the images are read straight
  into numpy, one `(H, W, 3) uint8` per camera at its own size. On an RTX 4090
  the room scene of design doc 16 (1.5M splats, six objects and a Panda arm,
  three 256×256 cameras) renders in about 6.6 ms with its meshes and shadow;
  a whole step through a local `world-engine serve` runs at 54–70 steps/s,
  against 4–6 on the browser path. The shaders ship in the package
  (`world_engine_client.wgsl`); a server never sends shader code.
- Meshes are drawn as the browser path's capture page draws them: three.js's
  MeshStandardMaterial (base colour, metalness, roughness, normal, AO and
  emissive maps, vertex colours, unlit and alpha-tested materials) from the
  WEM1 the engine compiles each object GLB into, robot links with three's
  painted finish, lit by the scene's own lighting baked at its light probe
  (or its HDRI, or three's room environment when the probe is dark), with
  design doc 14's local offsets, proxy boxes and scrolling belts. Renderer D
  adds the key light's shadow, which the capture page does not draw.
  Splats and meshes are blended above 8 bits (rgba16float on a GPU,
  rgba32float on a CPU adapter) and rounded to 8 bits once.
- How close: on design doc 16's room scene the images differ from the browser
  path's by 1.7/255 on average in the splat region and 0.9/255 where a mesh
  is in front (0.5/255 without the shadow); over nine test scenes the mesh
  region stays within 1.1/255 of the capture page. An RTX 4090, Mesa's
  lavapipe and the bundled SwiftShader differ by at most 0.1/255 in the
  splat region and 0.6/255 where a mesh is, and the same state gives the
  same pixels every run. To keep the browser path's images, pass
  `render_options={"renderer": "chrome"}`.
- Connect offers `render_protocols: [1, 2]` and `pipelines: ["wgsl/1"]`. A
  server from before renderer D ignores them and answers protocol 1: the
  browser path, as before. So does a newer server for a scene the pipeline
  cannot draw as the capture page does (a SuperSplat compressed PLY, an
  OpenEXR HDRI compressed with PIZ, PXR24, B44 or DWA, an object whose GLB
  did not compile to WEM1, an object whose GLB has GPU instancing, lines or
  points, a skin, morph targets, its own lights or compressed geometry, a
  lighting asset); it says why (`protocol_reason`), and the client passes
  that on in a `RuntimeWarning` and in `info["render"]["reason"]`.
- A session of renderer D downloads the assets it draws: the splat, the
  objects' WEM1, the robot links and an HDRI. The objects' GLBs, which only
  the browser path reads, stay on the server.
- No GPU is no error. The client looks for a GPU first and then for a
  software adapter: on Linux the system's Vulkan, the bundled loader with the
  system's drivers, Mesa's lavapipe, the bundled SwiftShader; on Windows WARP.
  A software adapter renders with one `RuntimeWarning`, and rendering slower
  than real time (29.4 steps/s) after a warm-up with another, which names the
  adapter, its type and the rate. When nothing renders, the browser path is
  tried, and then connect raises `RendererError` with the fixes. Each step
  of the search is probed in a subprocess that imports wgpu from where this
  process does, so a `pip install --user` or a `PYTHONPATH` install works.
- On a Linux machine that shows a GPU (a `/dev/dri` render node, NVIDIA's
  device) that Vulkan does not reach, for instance a GPU without its Vulkan
  driver (`mesa-vulkan-drivers`), renderer D would draw on the CPU. Headless
  Chrome then often reaches the GPU through OpenGL and is much faster (with
  Vulkan hidden on an RTX 4090 machine: about 15 steps/s against 0.9), so the
  client starts Chrome with hardware WebGL only and, when that works, draws
  with it, with one `RuntimeWarning` that says how to install the Vulkan
  driver. When Chrome renders in software too, renderer D draws, and that is
  remembered for a day. `render_options={"renderer": "wgsl"}` or an
  `adapter` of `"software"` or `"name:..."` keeps renderer D.
- A scene renderer D cannot draw (a PLY, WEM1 or HDRI it does not read, an
  object the server sent without its WEM1 or with content three draws and D
  does not, more splats than the GPU holds) is drawn by the browser path,
  with a `RuntimeWarning`, where Playwright is installed; `renderer="wgsl"`
  raises instead.
- On Windows renderer D draws through D3D12 with FXC, the HLSL compiler
  Windows has (wgpu uses DXC only when asked), on a GPU and on WARP; CI runs
  the client's tests on WARP. When renderer D fails on the adapter it opened
  (a shader the backend's compiler rejects, a validation error), the client
  takes the next adapter of the search, and the browser path after the last,
  with one `RuntimeWarning` that names the adapter and the error; the search
  passes over that adapter for the rest of the process.
- The `render` extra now brings `wgpu==0.32.0` and, on Linux x86_64 and
  aarch64, `world-engine-swiftshader==0.1.0`. Renderer D adds 10.7 MB to an
  install that has numpy and Pillow on x86_64, 16.4 MB on aarch64.
  Playwright and websockets stay in the extra for the browser path, so the
  whole extra is about 60 MB on x86_64 and 65 MB on aarch64.
- New package `world-engine-swiftshader` 0.1.0, built and published from
  this repository (`swiftshader/`, `.github/workflows/publish-swiftshader.yml`):
  a software Vulkan for Linux, SwiftShader and the Khronos Vulkan loader built
  from source, with their licences. Wheels for x86_64 (2 MB) and aarch64
  (about 8 MB), glibc 2.28 and later. A machine with a GPU does not use it.
- New `render_options`: `renderer` (`"auto"`, `"wgsl"`, `"chrome"`; default
  `WE_RENDERER`), `adapter` (`"auto"`, `"low-power"`, `"software"`,
  `"name:<part of the name>"`; default `WE_RENDER_ADAPTER`) and
  `color_format`. `info["render"]` names the renderer, the adapter of each
  step and the lighting.
- Renderer D decrypts the platform's assets into memory and uploads them to
  the GPU; no plain copy is written.
- The browser path's own browser, kept as the fallback of renderer D, follows
  design doc 16 §8.1. Playwright is pinned in the `render` extra
  (`playwright==1.60.0`, the engine's version), so every install downloads
  the same headless Chromium build, Chrome Headless Shell 148.
- Only the headless shell downloads (`playwright install --only-shell
  chromium`): about 120 MB on Linux and Windows and 100 MB on macOS, where
  Playwright's full Chromium and the shell came to 300 MB. It goes into a
  directory of this package next to the render cache,
  `<user cache dir>/digiarena/browsers` (`DIGIARENA_BROWSERS_DIR` moves it).
  Playwright's own browser directory is neither used nor changed. An
  installed Google Chrome is still used first.
- The download starts in the background when a browser-path renderer starts,
  while the scene downloads, and prints nothing on the terminal: neither the
  old "downloading Chromium" line nor Playwright's progress. It is logged to
  the `world_engine_client.render` logger. `python -m
  world_engine_client.render install` downloads it by hand; `... where` shows
  what is there.
- A failed download is tried once per renderer start. Its error reaches the
  caller with the fixes (the command above, a proxy or a download mirror,
  `images=False`), where a generic "no Chrome configuration rendered a frame"
  without fixes came before. So do the fixes for a missing browser and for
  missing system libraries; the latter is now `python -m playwright
  install-deps chromium`, which installs no browser.
- `WE_RENDER_BROWSER=chrome` or `WE_RENDER_BROWSER=shell` makes the browser
  path use only an installed Google Chrome or only the headless shell. Any
  other value stops `connect()` with a `RendererError` (code `bad_config`)
  that says how to fix it; the check of Chrome on the GPU does not keep it
  as "no Chrome on the GPU". `info["render"]` of the browser path names the
  browser and its version.
- `bench/igpu/` in the repository (not in the package or the sdist): a
  portable bench of renderer D against the browser path, for integrated GPUs
  on macOS, Linux and Windows. The engine's capture page that its
  browser-path runs need is not in the repository: it takes the build as a
  directory or a zip (`--capture-page`) and checks it against
  `capture_page/MANIFEST.json`. It runs the client of its own checkout,
  judges the speed only where both paths ran on the same GPU, and calls a
  result `PASS` only when every run worked. See its README.
- In a scene with several robots, `action_space` keeps every robot's gripper
  dim within ±1, by the server's rule: the mode's `gripper_indices`, else its
  `gripper_index`, else the last dim. 0.4.2 to 0.6.0 bounded only the last
  dim, so in `osc_pose_abs` a sampled action could fail the server's check
  and Gymnasium's `check_env` failed. The spaces of a scene with one robot do
  not change.
- The README and `connect()`'s docstring give the sum of an episode's rewards
  when a staged task shapes by distance. In points (`normalize_reward=False`)
  it is `earned - penalty + shaping["return"]`, and as the server sends them
  `(earned - penalty + shaping["return"]) / max`. They said `earned - penalty`.

## 0.6.0

- Reads the DigiArena platform's encrypted scene assets (DAC1, AES-256-GCM).
  A connect with images now says `asset_encryption: ["dac1"]`. 0.5.0 and
  earlier get `client_upgrade_required` from the platform instead of images.
- Scene assets go to a render cache that lasts between sessions:
  `<user cache dir>/digiarena/render/`, or `DIGIARENA_CACHE_DIR`, or
  `render_options={"cache_dir": ...}`. `close()` no longer deletes the
  cache, and there is no temporary directory any more. The platform's assets
  stay encrypted on disk, as they arrived. The key comes from the platform at
  every connect and lives in memory only, so a scene made private, a revoked
  key or a new key epoch makes the cached files unreadable at once. Robot
  meshes, the capture page's scripts and the scenes of a local
  `world-engine serve` are kept plain.
- Retention: files unused for 30 days are deleted unless their scene is
  pinned. `world_engine_client.cache` has `list`, `pin`, `unpin`, `clear`,
  `configure` (`max_age_days`, 0 keeps everything; `max_size_gb`) and
  `settings`. The same from a shell: `python -m world_engine_client.cache`.
  `DIGIARENA_CACHE_MAX_AGE_DAYS` and `DIGIARENA_CACHE_MAX_SIZE_GB` override
  the stored settings. The check runs when a renderer starts, at most once a
  day, after the session's own files are in place.
- A revoked key stops images: connect raises `RendererError` with code
  `asset_key_revoked`, even when the scene is cached.
- Fixed (BUG-19): a cached file was used without a check. Now every file is
  checked when a session uses it, and a damaged one is downloaded again.
- Fixed (BUG-19): processes sharing a `cache_dir` wrote the same `.part`
  file. One of them failed after the 300 s start limit, or the file on disk
  mixed two downloads and later sessions drew a broken scene. Now each
  download has its own temporary file, one process downloads while the
  others wait, and a process killed during its download releases the file at
  once.
- Fixed (BUG-19): a digest that was not 64 lowercase hexadecimal characters
  named a file anyway, `../` included. Such a session now fails before any
  download.
- Fixed (BUG-19): a `cache_dir` was created 0755 with files 0644. The cache
  is 0700 and its files 0600.
- Fixed (BUG-19): the capture page's scripts (5.5 MB) were downloaded at
  every connect. Scripts with Vite's content hash in their name are kept,
  per server; `index.html` is still fetched each time.
- Fixed (BUG-19): a killed process left plain scene assets in
  `/tmp/we_render_*`. The renderer deletes such directories once nobody has
  used them for a day; `python -m world_engine_client.cache --dir <old
  cache_dir> clear --legacy` deletes those (except ones used in the last
  hour) and the plain files of an old `cache_dir`. This version does not
  read those files.
- Fixed (BUG-32): any process on the machine could read the scene from the
  loopback HTTP server, and could connect to the loopback WebSocket and put
  its own images into the observations. Both now need a random token of the
  session, which only the page gets. Others get 403. The token does not
  appear in errors or logs.
- Assets and the capture page's scripts are downloaded before Chrome
  starts, and the download no longer counts against the renderer's start
  limit. A broken download resumes with a byte range, up to 3 times; a
  server that sends nothing for 60 s fails it.
- Fixed: servers close keep-alive connections that stay idle (Node after
  5 s), and the next request went out on the dead socket. A step or reset
  then raised `RecoveryRequiredError` although it never reached the server,
  and an observe raised `ProtocolError`. A renderer start longer than that
  did it to the first reset, which the end-to-end test showed with software
  rendering. The client now opens a new connection first when the server has
  closed the old one; it still never sends a request twice.
- The loopback server answers with `Cache-Control: no-store`. For the page
  it fetches only `index.html` and the files of the page's build from the
  server.
- The `render` extra adds `cryptography`. Only the platform's encrypted
  assets need it: without it, a session with a local `world-engine serve`
  still renders, and the client does not say that it reads DAC1.
- `RendererError` is defined in `world_engine_client.errors` (still
  importable from `world_engine_client.render`). New: `RenderFetchError`,
  an error status from the server's `/render/` routes.
- New: `connect(..., normalize_reward=True)`. With `normalize_reward=False`, a
  task with a `staged` reward returns its rewards in points: each reward times
  `info["task_progress"]["reward"]["max"]`, the task's total points. A step pays
  the points it earned, less the failure penalty on the step that fails, so the
  rewards of an episode add up to `earned - penalty` of its last
  `info["task_progress"]["reward"]`. With partial credit on, a step that gives
  partial credit back has a negative reward, though nothing failed. `sparse` and
  `dense` rewards come back unchanged, and so does every reward with the
  default. The server still sends normalized rewards, so older clients see no
  change.
- `normalize_reward` takes `True` or `False` only. Anything else raises
  `TypeError` before any request.
- With `normalize_reward=False`, a staged step whose info has no usable
  `task_progress.reward.max` raises `ProtocolError`, and the next step needs a
  `reset()`. Servers with staged rewards always send it.
- The README has a new section on rewards.

## 0.5.0

- Python 3.11 or newer is required. Python 3.10 reaches its end of life in
  October 2026. On 3.10, asyncio and concurrent.futures raise their own
  TimeoutError classes, which the renderer's `except TimeoutError` did not
  catch: a GPU path that hung on its first frame stopped the connect instead
  of moving on to the next Chrome flag set. In 3.11 both are the built-in
  TimeoutError.
- Camera images no longer need a GPU. Without hardware WebGL, the client
  renders in software (SwiftShader) and issues one `RuntimeWarning` before the
  first frame. `render_options={"allow_software": False}` restores the old
  behaviour and raises `RendererError` instead. `allow_software` now defaults
  to True in `LocalRenderer` and `candidate_args()`.
- A software first frame now gets the renderer's `timeout` (300 s by default).
  Before, every first frame had 15 s, a limit meant to catch a GPU path that
  hangs. On four CPU cores a 1.6M-splat scene needs about 27 s for its first
  frame, so connecting failed. Hardware adapters keep the 15 s limit.
- The renderer start keeps the 0.4.2 limit of `timeout` + 30 s (330 s by
  default) for launching Chrome and loading the scene, now shared by all
  Chrome flag sets. A software first frame gets `timeout` more. Downloading
  Chromium no longer counts against the limit. A scene that never finishes
  loading fails after about 330 s, as in 0.4.2, and the error lists what each
  flag set hit.
- Fixed: heartbeats now start at connect, before the renderer. A renderer
  start longer than the server's 90 s controller lease (a Chromium download on
  first use, or a large scene rendered in software) lost the session:
  `connect()` returned, and the first `reset()` raised `ProtocolError` ("no
  controller session; call connect first").
- New speed warning for hardware adapters. The client times the rendering of
  10 steps after the first 2, and if the median is below 3 steps per second
  it issues one `RuntimeWarning` per process. The warning names the adapter
  and the rate, and suggests `images=False` or fewer cameras. Basis: on an
  RTX 4090 with three 256×256 cameras, this renderer drew 0.4M–1.6M splat
  scenes at 6.8–17.2 steps per second (render only, measured 2026-10-26). 3 is
  about half of the slowest of these. Real time (29.4 steps per second) is not
  used, because this renderer stays below it even on an RTX 4090.
- Windows' WARP adapter ("Microsoft Basic Render Driver") counts as software.
- A capture page that throws before it opens its socket, for example because
  Chrome has no WebGL at all with a flag set, now fails that flag set after
  5 s instead of after the whole timeout.
- `LocalRenderer.close()` now closes its event loop.
- CI tests Python 3.11 and 3.12, and adds a job that installs the `render`
  extra on each and renders in a real headless Chrome.

Earlier releases are described in the commit history and the release tags.
