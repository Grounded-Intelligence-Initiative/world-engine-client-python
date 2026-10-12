"""wire.decode accepts exactly the documented packet and nothing else."""

import json
import sys

import numpy as np
import pytest
from conftest import jpeg, pack

from world_engine_client import ProtocolError, wire

CAMERAS = {"front": {"width": 8, "height": 6}, "wrist": {"width": 4, "height": 4}}
HEADER = {"obs": {"robot0_eef_pos": [0, 0, 0]}, "info": {"step_id": 1}}


def chunks():
    return [("front", 8, 6, jpeg(8, 6)), ("wrist", 4, 4, jpeg(4, 4, (10, 10, 240)))]


def test_round_trip_returns_rgb_uint8_per_camera():
    reply = wire.decode(pack(HEADER, chunks()), CAMERAS)
    rgb = reply["obs"]["rgb"]
    assert list(rgb) == ["front", "wrist"]
    assert rgb["front"].shape == (6, 8, 3) and rgb["front"].dtype == np.uint8
    assert rgb["wrist"].shape == (4, 4, 3)
    # Channel order survives: the first camera is red, the second blue.
    assert rgb["front"][0, 0, 0] > 150 > rgb["front"][0, 0, 2]
    assert rgb["wrist"][0, 0, 2] > 150 > rgb["wrist"][0, 0, 0]
    assert reply["info"] == {"step_id": 1}


def raw(header: bytes, payload: bytes = b"") -> bytes:
    return len(header).to_bytes(4, "little") + header + payload


def respec(**change):
    """A two-camera packet whose first descriptor is altered."""
    parts = chunks()
    specs, offset = [], 0
    for name, w, h, data in parts:
        specs.append({"name": name, "w": w, "h": h, "offset": offset, "length": len(data)})
        offset += len(data)
    specs[0].update(change)
    return pack(HEADER, parts, specs)


@pytest.mark.parametrize(
    "packet",
    [
        b"",
        b"\x01\x00",
        raw(b""),  # zero metadata length
        (2 * 1024 * 1024).to_bytes(4, "little") + b"{}",  # metadata longer than the packet
        raw(b"{not json"),
        raw(json.dumps({"wire_version": 3, "obs": {}}).encode()),  # no images key
        raw(json.dumps({"wire_version": 3, "images": []}).encode()),  # no obs, missing cameras
        raw(json.dumps([1, 2]).encode()),
    ],
    ids=["empty", "short", "zero-meta", "meta-overrun", "bad-json", "no-images", "no-obs", "list"],
)
def test_malformed_envelopes_are_rejected(packet):
    with pytest.raises(ProtocolError, match="malformed observation"):
        wire.decode(packet, CAMERAS)


@pytest.mark.parametrize("version", [2, 4, "3", 3.0, True, None])
def test_only_integer_wire_version_3_is_accepted(version):
    meta = json.dumps({"wire_version": version, "obs": {}, "images": []}).encode()
    with pytest.raises(ProtocolError):
        wire.decode(raw(meta), {})


@pytest.mark.parametrize(
    "change",
    [
        {"name": "wrist"},  # duplicate
        {"name": "side"},  # not negotiated
        {"length": 0},
        {"length": -5},
        {"length": "12"},
        {"length": 12.0},
        {"offset": 1},
        {"offset": "0"},
        {"w": 8.0},
        {"h": "6"},
        {"w": 16},  # disagrees with the negotiated dimensions
        {"length": 10**9},  # beyond the payload
    ],
    ids=lambda c: "-".join(f"{k}={v!r}" for k, v in c.items()),
)
def test_bad_image_descriptors_are_rejected(change):
    with pytest.raises(ProtocolError):
        wire.decode(respec(**change), CAMERAS)


def test_missing_camera_and_trailing_bytes_are_rejected():
    with pytest.raises(ProtocolError, match="missing images"):
        wire.decode(pack(HEADER, chunks()[:1]), CAMERAS)
    with pytest.raises(ProtocolError, match="trailing"):
        wire.decode(pack(HEADER, chunks()) + b"\x00", CAMERAS)


def test_non_contiguous_offsets_are_rejected():
    parts = chunks()
    specs = [
        {"name": "front", "w": 8, "h": 6, "offset": 0, "length": len(parts[0][3])},
        {"name": "wrist", "w": 4, "h": 4, "offset": len(parts[0][3]) + 1, "length": 1},
    ]
    with pytest.raises(ProtocolError, match="offset"):
        wire.decode(pack(HEADER, parts, specs), CAMERAS)


def test_truncated_and_non_jpeg_images_are_rejected():
    one = {"front": CAMERAS["front"]}
    good = jpeg(8, 6)
    with pytest.raises(ProtocolError, match="truncated JPEG"):
        wire.decode(pack(HEADER, [("front", 8, 6, good[:-2])]), one)
    with pytest.raises(ProtocolError):
        wire.decode(pack(HEADER, [("front", 8, 6, b"not an image" + b"\xff\xd9")]), one)


def test_decoded_size_must_match_descriptor():
    # The descriptor and the negotiated camera agree, but the JPEG itself is 4x4.
    packet = pack(HEADER, [("front", 8, 6, jpeg(4, 4))])
    with pytest.raises(ProtocolError, match="dimensions or format"):
        wire.decode(packet, {"front": CAMERAS["front"]})


def test_response_size_limit(monkeypatch):
    packet = pack(HEADER, chunks())
    monkeypatch.setattr(wire, "MAX_RESPONSE", len(packet) - 1)
    with pytest.raises(ProtocolError, match="packet length"):
        wire.decode(packet, CAMERAS)


def test_missing_pillow_is_an_actionable_import_error(monkeypatch):
    packet = pack(HEADER, chunks())
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(ImportError, match=r"world-engine-client\[img\]"):
        wire.decode(packet, CAMERAS)
