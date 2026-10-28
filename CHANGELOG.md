# Changelog

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
