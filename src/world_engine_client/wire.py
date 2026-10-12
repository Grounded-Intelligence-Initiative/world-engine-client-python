"""Strict protocol-3 binary decoder; Pillow is imported only for images."""

import io
import json

import numpy as np

from .errors import ProtocolError

CONTENT_TYPE = "application/vnd.world-engine.observation-v3"
MAX_RESPONSE = 64 * 1024 * 1024


def decode(data: bytes, cameras: dict) -> dict:
    try:
        if len(data) < 4 or len(data) > MAX_RESPONSE:
            raise ValueError("invalid packet length")
        length = int.from_bytes(data[:4], "little")
        if not 0 < length <= min(len(data) - 4, 1024 * 1024):
            raise ValueError("invalid metadata length")
        header = json.loads(data[4 : 4 + length])
        if type(header["wire_version"]) is not int or header["wire_version"] != 3:
            raise ValueError("unsupported observation version")
        payload = data[4 + length :]
        seen, offset = set(), 0
        for spec in header["images"]:
            name = spec["name"]
            if name in seen or name not in cameras:
                raise ValueError("unexpected or duplicate camera")
            seen.add(name)
            n = spec["length"]
            if (
                type(n) is not int
                or type(spec["offset"]) is not int
                or type(spec["w"]) is not int
                or type(spec["h"]) is not int
                or n <= 0
                or spec["offset"] != offset
                or offset + n > len(payload)
            ):
                raise ValueError("invalid image offset/length")
            expected = cameras[name]
            if (spec["w"], spec["h"]) != (expected["width"], expected["height"]):
                raise ValueError("camera dimensions changed")
            offset += n
        if seen != set(cameras) or offset != len(payload):
            raise ValueError("missing images or trailing data")
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "RGB observations require pip install 'world-engine-client[img]'"
            ) from exc
        rgb = {}
        for spec in header["images"]:
            chunk = payload[spec["offset"] : spec["offset"] + spec["length"]]
            if not chunk.endswith(b"\xff\xd9"):
                raise ValueError("truncated JPEG")
            with Image.open(io.BytesIO(chunk)) as img:
                if img.format != "JPEG" or img.size != (spec["w"], spec["h"]):
                    raise ValueError("decoded image dimensions or format mismatch")
                rgb[spec["name"]] = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
        header["obs"]["rgb"] = rgb
        return header
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ProtocolError(f"malformed observation: {exc}") from exc
