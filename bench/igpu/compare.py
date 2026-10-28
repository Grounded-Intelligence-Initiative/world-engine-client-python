"""How far two sets of camera images are apart: the M-D1 gate numbers of design
doc 16 §9.6 (the engine's scripts/wgsl/compare_images.py), in 0-255 units over
the three channels.

The bench scenes have no meshes, so the splat region (pixels with no mesh in
front) is the whole image. ``mean`` is |difference| pooled over every pixel
and channel of every image (the gate number); ``worst_image`` the largest
per-image mean; ``signed_mean`` the test image minus the reference; ``max``
the largest single difference; ``frac_gt2`` and ``frac_gt16`` the share of
pixels whose largest channel difference exceeds 2 and 16.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def load_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def save_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(path, optimize=True)


def digest(image: np.ndarray) -> str:
    """The SHA-256 of an image's pixels, to see whether runs agree exactly."""
    return hashlib.sha256(np.ascontiguousarray(image, dtype=np.uint8).tobytes()).hexdigest()


def compare(refs: dict[str, np.ndarray], tests: dict[str, np.ndarray]) -> dict:
    """``refs`` and ``tests`` map image names to (H, W, 3) uint8; only names in
    both count. Missing names are listed."""
    total = signed = 0.0
    values = pixels = gt2 = gt16 = 0
    biggest = 0
    per_image = {}
    for name in sorted(refs):
        if name not in tests:
            continue
        ref = refs[name].astype(np.int16)
        test = np.asarray(tests[name]).astype(np.int16)
        if ref.shape != test.shape:
            raise ValueError(f"{name}: image {test.shape} against reference {ref.shape}")
        d = test - ref
        a = np.abs(d)
        per_pixel = a.max(-1)
        total += float(a.sum())
        signed += float(d.sum())
        values += a.size
        pixels += per_pixel.size
        biggest = max(biggest, int(a.max()))
        gt2 += int((per_pixel > 2).sum())
        gt16 += int((per_pixel > 16).sum())
        per_image[name] = round(float(a.mean()), 4)
    if not values:
        return {"images": 0, "missing": sorted(refs)}
    return {
        "images": len(per_image),
        "mean": round(total / values, 4),
        "worst_image": max(per_image.values()),
        "signed_mean": round(signed / values, 4),
        "max": biggest,
        "frac_gt2": round(gt2 / pixels, 5),
        "frac_gt16": round(gt16 / pixels, 5),
        "per_image": per_image,
        "missing": sorted(set(refs) - set(tests)),
    }
