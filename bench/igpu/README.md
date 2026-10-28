# Integrated-GPU bench: renderer D against the browser path

This bench answers, for the GPU of the machine it runs on: does renderer D
(`WGSLRenderer`, wgpu) draw what it draws on an RTX 4090, and is it at least
as fast as the browser path (`LocalRenderer`, headless Chrome running the
engine's capture page with Spark)? Design doc 16 and task book §8.1 item 4 (b)
need the answer for integrated GPUs before Playwright can leave the `render`
extra; the development machine has none, so the bench is portable: macOS
(Metal), Linux (Vulkan) and Windows (D3D12, or Vulkan where the driver has it).

It needs no engine and no network after the first run: the scenes are made on
the machine, byte for byte the same everywhere, and a local stand-in of a
server's `/render/` route hands both renderers the same assets and the same
capture page. The capture page is the engine's build, which is not in this
repository (see "The capture page" below).

## What runs

Two splat-only scenes, the same for both paths, seen by the same three 256×256
cameras through the same 221 states (`trace/`: r2_400k's trace, 21 warm-up and
200 timed states; the wrist camera moves, the other two stand still):

- r2_400k: design doc 16's 400k-splat scene without its meshes and robot.
  It is timed. Its splat is a uniform grey fog: every pixel of every image
  saturates to the same grey, so its images only show whether a renderer
  draws at all.
- room200k: a box room of 200k flat splats (a checkered floor, four banded
  walls, two striped boxes, translucent floaters), SH degree 3. Its images are
  the quality test.

`scene.py` makes both splats and checks their SHA-256: r2_400k's with the
generator it was made with in the first place (DigiArena website,
`scripts/e2e_incremental_save.py`), room200k's with exact arithmetic only, so
neither depends on the platform. Neither has meshes, so every pixel is splat
region (design doc 16 §9.10 item 5).

A round runs each path once, each in a fresh process (`worker.py`), and the
order of the paths turns from round to round. The paths (`--paths`):

| key | what |
|---|---|
| `d` | renderer D with its default colour target (rgba16float on a GPU, rgba32float on a CPU adapter) |
| `chrome` | the browser path as users get it: images JPEG-encoded by the page |
| `d_rgba8` | renderer D drawing into rgba8unorm, to see what the default target costs here |

A run starts the renderer, renders the first state and up to 20 warm-up states
(10 s at most), times up to 200 states (60 s at most, 5 at least), then renders
the reference states of both scenes and compares them with `refs/`. After the
rounds, one more browser-path run (`chrome_png`) reads its images back
losslessly, PNG in place of JPEG, to compare Spark here with Spark on the RTX
4090. Speed is render only: the recorded trace replayed, no server, no physics:
what drawing the cameras of one gym step costs.

The browser path uses the headless shell of the playwright that the `render`
extra pins (Chrome Headless Shell 148), downloaded once into
`<user cache dir>/digiarena/browsers` on the first run, so every machine runs
the same browser build. `--browser auto` uses an installed Google Chrome first,
as the client does by default.

## The capture page

The browser path runs the engine's capture page: a production build of the
World Engine frontend at engine commit 1af5543 (production on 2026-10-28),
`index.html` and the two scripts it loads. The engine's repository is
private, so the build is not committed here: `capture_page/MANIFEST.json`
names its files and their SHA-256, and the bench checks every file before a
run. It takes the build as a directory or as the zip
`igpu-capture-page-1af5543.zip` (1.9 MB), from `--capture-page DIR|ZIP`, the
environment variable `IGPU_BENCH_CAPTURE_PAGE`, or `capture_page/` itself
(git ignores the build there). `pack_capture_page.py <engine>/frontend/dist`
makes the zip from an engine checkout built at that commit
(`npm run build`). Runs of renderer D alone (`--paths d`) need no page.

## Install and run

Python 3.11 or later, and world-engine-client 0.7.0 or later, the first
release with the browser path this bench expects, with its `render` extra
(wgpu, Pillow, cryptography, the pinned playwright, websockets, and on Linux
world-engine-swiftshader):

```bash
python -m venv bench-venv && . bench-venv/bin/activate   # Windows: bench-venv\Scripts\activate
pip install 'world-engine-client[render]'
git clone https://github.com/Grounded-Intelligence-Initiative/world-engine-client-python
cd world-engine-client-python
python bench/igpu/run.py --capture-page ~/Downloads/igpu-capture-page-1af5543.zip --label "M4 Pro, on power"
```

On Linux the headless shell may need Chromium's system libraries once:
`python -m playwright install-deps chromium` (needs sudo).

It takes 5 to 15 minutes on a GPU. It prints each run's adapter, speed and
quality, and writes `igpu_bench_<host>_<time>.json` (`--out` names it) and the
images of every run next to it (`--no-images` keeps none). Send the JSON. The
render cache of the runs (about 80 MB: the scenes in plaintext, the page) is
in a temporary directory that the bench removes at the end (`--keep-cache`
keeps it).

## Making both paths use the integrated GPU

The bench prints the adapter of every run: renderer D's name, type
(`IntegratedGPU`, `DiscreteGPU`, `CPU`) and backend; Chrome's WebGL renderer
string. `summary.same_gpu` in the JSON compares the two by the device's name
(the WebGL string's device, without "(R)", PCI ids and ANGLE's backend
words), so an integrated and a discrete GPU of one vendor count as two; any
CPU renderer on both sides (lavapipe, SwiftShader, WARP) counts as the one
CPU. At the end the bench prints both names and whether they are the same.

- Renderer D: `--adapter name:Intel` takes the adapter whose name contains
  "Intel"; `--adapter low-power` takes an integrated GPU before a discrete one.
  `WE_RENDER_ADAPTER` does the same. To list the adapters:
  `python -c "import wgpu; [print(a.info) for a in wgpu.gpu.enumerate_adapters_sync()]"`.
  On Windows the same GPU can show up twice, once for Vulkan and once for
  D3D12; renderer D takes the first that renders (in wgpu's order, Vulkan
  first), and the JSON says which. To time D3D12,
  `--d-env "VK_LOADER_DRIVERS_DISABLE=*"` hides the Vulkan drivers from
  renderer D's runs (not tried yet).
- Chrome on Windows: Settings → System → Display → Graphics: add
  `%LOCALAPPDATA%\digiarena\browsers\chromium_headless_shell-1223\chrome-headless-shell-win64\chrome-headless-shell.exe`
  (it is there after the first run, or after
  `python -m world_engine_client.render install`) and set it to "Power
  saving". Its WebGL string should then name the Intel GPU.
- Chrome on Linux with an NVIDIA GPU too: leave `__NV_PRIME_RENDER_OFFLOAD`
  and `__GLX_VENDOR_LIBRARY_NAME` unset; if Chrome still names NVIDIA,
  `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json`
  makes its EGL use Mesa.
- macOS: one GPU; nothing to choose.

Watch the GPU while it runs (`intel_gpu_top`, Task Manager → Performance,
Activity Monitor → GPU History) to see the integrated GPU working and the
discrete one idle.

## Thresholds

`summary.checks` in the JSON:

| check | limit |
|---|---|
| room200k: renderer D against renderer D on the RTX 4090, mean \|diff\| | ≤ 1.5/255 |
| room200k: renderer D against Spark (lossless) on the RTX 4090, mean \|diff\| | ≤ 3/255 |
| r2_400k: renderer D against Spark on the RTX 4090 (only: does D draw) | ≤ 3/255 |
| r2_400k: renderer D's median steps/s over the browser path's, same GPU | ≥ 1 |

The first two are the gates of design doc 16 §9.4 and §9.6 (M-D1): across GPU
vendors 1.5/255 in the splat region with the RTX 4090 as reference, against
the editor's Spark 3/255. The mean is |difference| over all pixels and
channels of the five reference images (`compare.py`); each run's mean counts,
the worst run decides. The last is task book §8.1 item 4 (b): on each kind of
adapter renderer D is not slower than the browser path. When one fails, task
book §8.1 item 4 says what follows: for quality, look for the cause first (the
adapter's precision, the driver); if it stays above the limit, that kind of
adapter keeps the browser path as its default. For speed, that kind of adapter
defaults to the faster path.

`summary.verdict` (printed last) is `PASS` only when every planned run worked
and all four checks were judged and passed, `FAIL` when a check failed, and
`INCOMPLETE` otherwise, with `summary.incomplete_because` saying why: a run
that failed (its error is in `runs`), a run that never ran, or a check that
could not be judged. The speed check is judged only when both paths ran on
the same GPU; on different GPUs their ratio says nothing about either, and
the check says so. An `INCOMPLETE` result does not answer task book §8.1
item 4 (b) for that machine: fix what it names and run again, or send it as
it is. The exit status says whether the bench ran, not the verdict.

## The result file

One JSON object:

- `machine`: `os`, `cpu` (model, cores, memory, load), `gpus` (every adapter
  wgpu sees, with its driver in `description`, and the system's own list:
  `lspci`, `nvidia-smi` and `vulkaninfo --summary` on Linux,
  `system_profiler` on macOS, `Win32_VideoController` on Windows), `power`,
  `python`, `packages` (world-engine-client, wgpu, playwright and the rest),
  `browsers` (where the headless shell is), `environment` (the variables
  that pick a GPU).
- `config`, `capture_page` (its manifest), `started`, `finished`.
- `runs`: one per run: `key`, `round`, `adapter` (renderer D: name, type,
  backend, the search step, colour target and the search's trail; Chrome:
  the WebGL renderer string, the browser and its version, the flags),
  `startup_s`, `first_frame_s`, `timed` (frames, steps/s, frame time median
  and spread), `quality` (per scene and reference: mean, worst image, signed
  mean, largest difference, share of pixels off by more than 2 and 16, and a
  digest of each image), `warnings` (every warning's text), `log`, the load
  average at its start and end (Linux and macOS), and on failure `error`
  (type, message, fixes, traceback) and `stderr_tail`.
- `summary`: per path the median and range of steps/s, the quality numbers,
  whether the images were the same in every round, the adapters and warnings;
  `d_over_chrome`, `same_gpu`, `failed_runs`, `missing_runs`, `checks`,
  `complete`, `incomplete_because`, `pass` (True, False or None) and
  `verdict`.

## Numbers on the development machine

`make_refs.py` made `refs/` on an RTX 4090 (driver 580.178.04): renderer D's
images (rgba16float) and Spark's (PNG readback through the same capture
page). Both rendered twice the same. Renderer D against Spark there: room200k
1.629/255, r2_400k 0.402/255.

The same bench there, 2026-10-28, three rounds (render only, three 256×256
cameras of r2_400k; only GPU0 visible). The machine, a Threadripper PRO
5975WX with 64 threads, was shared: the load average at the start of a run
was 5 to 34 for the RTX 4090 rows, 17 to 46 for lavapipe and 18 to 87 for
SwiftShader. The CPU rows force Chrome onto its SwiftShader
(`--chrome-args "--headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --no-sandbox"`)
and renderer D onto a CPU adapter (`--adapter software`; for the bundled
SwiftShader also `--d-env VK_ICD_FILENAMES=/nonexistent/none.json`, which
hides lavapipe):

| renderer D on | D, steps/s | Chrome, steps/s | D / Chrome | D rgba8unorm | room200k: D vs 4090 D | D vs Spark | Chrome (JPEG) vs Spark | Spark here vs 4090 Spark |
|---|---|---|---|---|---|---|---|---|
| RTX 4090 (Chrome: the 4090 through ANGLE GL ES) | 192.0 (188.8–197.2) | 17.44 (17.38–17.49) | 11.0 | 210.8 | 0.000 | 1.629 | 2.607 | 0.000 |
| lavapipe (Chrome: its SwiftShader) | 3.06 (2.46–3.17) | 0.081 (0.075–0.089) | 37.8 | 3.05 | 0.071 | 1.630 | 2.855 | 1.487 |
| bundled SwiftShader (Chrome: its SwiftShader) | 0.92 (0.91–0.92) | 0.078 (0.076–0.080) | 11.8 | 0.91 | 0.070 | 1.631 | 2.855 | 1.487 |

Medians over the rounds, ranges in brackets; D / Chrome is the ratio of the
medians. The quality columns are the mean |difference| over the five room200k
reference images, in 1/255. Every check passed on all three. Renderer D's
images were the same in every round on each adapter, and so were the browser
path's. A run two hours earlier, with the same measuring code (its records
kept less: no load average), gave the same quality numbers and a D / Chrome
of 10.7, 34.7 and 10.1.

## Files

- `run.py`: the bench. `worker.py`: one run. `scene.py`: the scenes, the
  trace, the `/render/` stand-in. `compare.py`: the image numbers.
  `machine.py`: the machine. `page.py`: finds and checks the capture page.
  `pack_capture_page.py`: makes its zip. `make_refs.py`: makes `refs/` again
  (on an RTX 4090 only).
- `trace/`: `scene_init.json`, `states.jsonl`, `meta.json`.
- `refs/<scene>/{d_rgba16float_rtx4090,spark_rtx4090}/`: the reference
  images; `refs/refs.json` says what made them.
- `capture_page/`: `MANIFEST.json` (the build's files and their SHA-256) and
  `LICENSES.md` (the notices of the libraries the build bundles; the zip
  carries both).
