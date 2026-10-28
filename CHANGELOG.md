# Changelog

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
