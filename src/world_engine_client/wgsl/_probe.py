"""One adapter probe of D's adapter search, run in its own process (adapters.py).

wgpu-py creates one wgpu instance per process, at the first enumeration, and
that instance keeps the Vulkan loader it loaded. So every probe of the search
runs here, in a fresh interpreter started with ``python -I``, and imports
nothing but wgpu and numpy:

    python -I -B _probe.py --out RESULT.json [--sys-path] [--preload LIBVULKAN]
        [--icd ICD.json] [--pick hardware|software|fallback|name:SUBSTR|...] [--prefer ...]

--sys-path read the module search path from stdin (a JSON list of strings) and
           search there first. ``-I`` leaves out PYTHONPATH and the user's
           site-packages, where pip puts wgpu when the interpreter's own
           site-packages is not writable (``pip install --user``, ``--target``):
           the search passes its process's sys.path, so the probe imports the
           wgpu and numpy that process imports.
--preload  ctypes.CDLL(path, RTLD_GLOBAL) before the first enumeration, so
           wgpu-native's dlopen("libvulkan.so.1") gets this loader
--icd      VK_ICD_FILENAMES and VK_DRIVER_FILES (the old and new names)
--pick     comma-separated kinds tried in order, the first that renders wins:
           hardware (a GPU on Vulkan, Metal or D3D12), software (a CPU adapter on
           those), fallback (request_adapter(force_fallback_adapter=True), WARP on
           Windows), name:SUBSTR (by device name)
--prefer   among hardware adapters: discrete (default) or integrated first

The result is written to --out as JSON, never to stdout: some Mesa builds print
NIR there on a shader-cache miss. ``import_error`` in it says that wgpu or numpy
did not import (then no adapter was looked at). It lists the adapters seen and, for each kind
tried, whether a device with the WebGPU default limits renders: a 256-wide
compute dispatch with workgroup memory and a barrier, and a blended draw into
the colour target the renderer would use (render_wgsl.default_color_format),
read back through an rgba8unorm resolve. This file must not import anything
from its package (it runs as a script).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import traceback

NATIVE = ("Vulkan", "Metal", "D3D12")
HARDWARE = ("DiscreteGPU", "IntegratedGPU")
MiB = 1 << 20
DEFAULT_LIMITS = {
    "max-storage-buffer-binding-size": 128 * MiB,
    "max-buffer-size": 256 * MiB,
    "max-compute-invocations-per-workgroup": 256,
    "max-compute-workgroup-size-x": 256,
    "max-compute-workgroup-storage-size": 16384,
    "max-storage-buffers-per-shader-stage": 8,
    "max-bind-groups": 4,
    "max-compute-workgroups-per-dimension": 65535,
}

COMPUTE = """
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
var<workgroup> s: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_id) l: vec3u) {
  s[l.x] = data[g.x];
  workgroupBarrier();
  data[g.x] = s[255u - l.x] * 2u;
}
"""
DRAW = """
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(p[i], 0.5, 1.0);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(0.4, 0.2, 0.1, 0.5); }
"""
RESOLVE = """
@group(0) @binding(0) var colour: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = textureLoad(colour, vec2i(pos.xy), 0);
  return vec4f(round(clamp(c.rgb, vec3f(0.0), vec3f(1.0)) * 255.0) / 255.0, 1.0);
}
"""


def info_of(a) -> dict:
    i = a.info
    return {
        k: i.get(k) for k in ("vendor", "device", "adapter_type", "backend_type", "description")
    }


def colour_format(adapter) -> str:
    # the same rule as render_wgsl.default_color_format
    if adapter.info.get("adapter_type") == "CPU":
        return "rgba32float" if "float32-blendable" in adapter.features else "rgb10a2unorm"
    return "rgba16float"


def verify(wgpu, np, adapter) -> dict:
    """Render with this adapter the way the renderer does; raise if it does not."""
    out: dict = {}
    t = time.perf_counter()
    for k, v in DEFAULT_LIMITS.items():
        have = adapter.limits.get(k)
        if have is not None and have < v:
            raise RuntimeError(f"limit {k} is {have}, below the WebGPU default {v}")
    fmt = colour_format(adapter)
    feats = ["float32-blendable"] if fmt == "rgba32float" else []
    dev = adapter.request_device_sync(required_limits=dict(DEFAULT_LIMITS), required_features=feats)
    out["device_ms"] = round((time.perf_counter() - t) * 1e3, 1)
    n = 1 << 14
    U = wgpu.BufferUsage
    buf = dev.create_buffer_with_data(
        data=np.arange(n, dtype=np.uint32), usage=U.STORAGE | U.COPY_SRC
    )
    pl = dev.create_compute_pipeline(
        layout="auto",
        compute={"module": dev.create_shader_module(code=COMPUTE), "entry_point": "main"},
    )
    bg = dev.create_bind_group(
        layout=pl.get_bind_group_layout(0), entries=[{"binding": 0, "resource": {"buffer": buf}}]
    )
    enc = dev.create_command_encoder()
    cp = enc.begin_compute_pass()
    cp.set_pipeline(pl)
    cp.set_bind_group(0, bg)
    cp.dispatch_workgroups(n // 256)
    cp.end()
    dev.queue.submit([enc.finish()])
    got = np.frombuffer(dev.queue.read_buffer(buf), np.uint32)
    want = (np.arange(n).reshape(-1, 256)[:, ::-1].ravel() * 2).astype(np.uint32)
    if not np.array_equal(got, want):
        raise RuntimeError("the compute check gave wrong values")
    W, H = 64, 48
    T = wgpu.TextureUsage
    blend = {
        "color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
        "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
    }
    sm = dev.create_shader_module(code=DRAW)
    draw = dev.create_render_pipeline(
        layout="auto",
        vertex={"module": sm, "entry_point": "vs"},
        primitive={"topology": "triangle-list"},
        fragment={"module": sm, "entry_point": "fs", "targets": [{"format": fmt, "blend": blend}]},
    )
    colour = dev.create_texture(
        size=(W, H, 1), format=fmt, usage=T.RENDER_ATTACHMENT | T.TEXTURE_BINDING
    )
    eight = dev.create_texture(
        size=(W, H, 1), format="rgba8unorm", usage=T.RENDER_ATTACHMENT | T.COPY_SRC
    )
    rm = dev.create_shader_module(code=RESOLVE)
    bgl = dev.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "unfilterable-float", "view_dimension": "2d"},
            }
        ]
    )
    resolve = dev.create_render_pipeline(
        layout=dev.create_pipeline_layout(bind_group_layouts=[bgl]),
        vertex={"module": rm, "entry_point": "vs"},
        primitive={"topology": "triangle-list"},
        fragment={"module": rm, "entry_point": "fs", "targets": [{"format": "rgba8unorm"}]},
    )
    rbg = dev.create_bind_group(
        layout=bgl, entries=[{"binding": 0, "resource": colour.create_view()}]
    )
    rb = dev.create_buffer(size=W * H * 4, usage=U.MAP_READ | U.COPY_DST)
    enc = dev.create_command_encoder()
    p = enc.begin_render_pass(
        color_attachments=[
            {
                "view": colour.create_view(),
                "load_op": "clear",
                "store_op": "store",
                "clear_value": (0.0, 0.0, 1.0, 1.0),
            }
        ]
    )
    p.set_pipeline(draw)
    p.draw(3)
    p.end()
    p = enc.begin_render_pass(
        color_attachments=[
            {
                "view": eight.create_view(),
                "load_op": "clear",
                "store_op": "store",
                "clear_value": (0, 0, 0, 1),
            }
        ]
    )
    p.set_pipeline(resolve)
    p.set_bind_group(0, rbg)
    p.draw(3)
    p.end()
    enc.copy_texture_to_buffer(
        {"texture": eight},
        {"buffer": rb, "offset": 0, "bytes_per_row": W * 4, "rows_per_image": H},
        (W, H, 1),
    )
    dev.queue.submit([enc.finish()])
    rb.map_sync(wgpu.MapMode.READ)
    px = np.frombuffer(rb.read_mapped(copy=True), np.uint8).reshape(H, W, 4)
    rb.unmap()
    # 0.4, 0.2, 0.1 + 0.5 * blue 1.0 -> (102, 51, 153)
    d = int(np.abs(px[:, :, :3].astype(int) - np.array([102, 51, 153])).max())
    if d > 2:
        raise RuntimeError(
            f"the draw check is off by {d}/255 (pixel {px[H // 2, W // 2].tolist()})"
        )
    out["color_format"] = fmt
    out["verify_ms"] = round((time.perf_counter() - t) * 1e3, 1)
    for b in (buf, rb):
        b.destroy()
    for tex in (colour, eight):
        tex.destroy()
    dev.destroy()
    return out


def use_sys_path(paths) -> None:
    """Search modules in ``paths`` (the starting process's sys.path) first,
    then in this interpreter's own entries."""
    given = [p for p in paths if isinstance(p, str)]
    sys.path[:] = given + [p for p in sys.path if p not in given]


def candidates(adapters, kind: str, prefer: str) -> list:
    ads = [a for a in adapters if a.info.get("backend_type") in NATIVE]
    if kind == "hardware":
        hw = [a for a in ads if a.info.get("adapter_type") in HARDWARE]
        first = "IntegratedGPU" if prefer == "integrated" else "DiscreteGPU"
        return sorted(hw, key=lambda a: a.info.get("adapter_type") != first)
    if kind == "software":
        return [a for a in ads if a.info.get("adapter_type") == "CPU"]
    if kind.startswith("name:"):
        s = kind[5:].lower()
        return [a for a in ads if s in (a.info.get("device") or "").lower()]
    raise ValueError(f"unknown kind {kind!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preload")
    ap.add_argument("--icd")
    ap.add_argument("--pick", default="hardware")
    ap.add_argument("--prefer", default="discrete")
    ap.add_argument("--sys-path", action="store_true")
    a = ap.parse_args(argv)
    res: dict = {"python": sys.version.split()[0], "platform": sys.platform, "pick": a.pick}
    t0 = time.perf_counter()
    try:
        if a.sys_path:
            use_sys_path(json.load(sys.stdin))
        if a.icd:
            os.environ["VK_ICD_FILENAMES"] = os.environ["VK_DRIVER_FILES"] = a.icd
        if a.preload:
            ctypes.CDLL(a.preload, mode=ctypes.RTLD_GLOBAL)
        elif sys.platform.startswith("linux"):
            # wgpu has no Vulkan adapters at all without a loader; say why
            try:
                ctypes.CDLL("libvulkan.so.1")
                res["loader"] = "system"
            except OSError as exc:
                res["loader"] = None
                res["loader_error"] = str(exc)[:300]
        import numpy as np
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        res["wgpu"] = wgpu.__version__
        res["numpy"] = np.__version__
        # Vulkan, Metal and D3D12 only: an OpenGL adapter has no compute.
        set_instance_extras(backends=["Primary"])
        adapters = wgpu.gpu.enumerate_adapters_sync()
        res["adapters"] = [info_of(x) for x in adapters]
        res["tried"] = []
        for kind in a.pick.split(","):
            if kind == "fallback":
                try:
                    found = [wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)]
                except Exception as exc:  # noqa: BLE001 - reported
                    res["tried"].append(
                        {"kind": kind, "error": f"{type(exc).__name__}: {exc}"[:500]}
                    )
                    continue
                found = [x for x in found if x is not None]
            else:
                found = candidates(adapters, kind, a.prefer)
            for ad in found:
                entry = {"kind": kind, "adapter": info_of(ad)}
                try:
                    entry.update(verify(wgpu, np, ad))
                    entry["ok"] = True
                except Exception as exc:  # noqa: BLE001 - reported
                    entry["ok"] = False
                    entry["error"] = f"{type(exc).__name__}: {exc}"[:800]
                res["tried"].append(entry)
                if entry["ok"]:
                    res["ok"] = True
                    res["picked"] = entry
                    break
            if res.get("ok"):
                break
            if not found:
                why = "no such adapter"
                if res.get("loader", "system") is None:
                    why = "no Vulkan loader (libvulkan.so.1) on this system"
                res["tried"].append({"kind": kind, "error": why})
        res.setdefault("ok", False)
    except Exception as exc:  # noqa: BLE001 - reported
        res["ok"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        res["traceback"] = traceback.format_exc()[-3000:]
        if isinstance(exc, ImportError):
            res["import_error"] = res["error"]
    res["seconds"] = round(time.perf_counter() - t0, 3)
    tmp = a.out + ".part"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
