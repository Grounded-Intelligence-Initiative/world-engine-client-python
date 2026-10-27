# Changelog

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
