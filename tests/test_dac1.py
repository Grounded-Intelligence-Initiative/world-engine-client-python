"""DAC1, the platform's encrypted scene asset format: the client against the
shared test vectors.

tests/data/dac1_vectors.json is the contract between the implementations:
the platform gateway (Node crypto, website tests/fixtures/dac1-vectors.json),
the browser (WebCrypto) and the engine's pure-Python reference (engine
tests/dac1_ref.py, copied here unchanged as the oracle). All copies hash to
VECTORS_SHA256; the engine and the website pin the same value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import dac1_ref as ref
import pytest

pytest.importorskip("cryptography")

from world_engine_client import dac1  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "data" / "dac1_vectors.json"
# The engine's tests/test_dac1_vectors.py and the website's tests/dac1.test.ts
# pin this hash of their copies.
VECTORS_SHA256 = "68fcd2668430004ad13fe2df8048d4f2f54d6582d31f67820ebbae7d80e1f49e"


def _vectors() -> dict:
    return json.loads(VECTORS.read_text())


def test_the_vectors_are_the_engines_and_the_websites():
    assert hashlib.sha256(VECTORS.read_bytes()).hexdigest() == VECTORS_SHA256
    assert ref.build_vectors() == _vectors(), "the copied reference still makes the vectors"


def test_keys_and_names_match_the_vectors():
    v = _vectors()
    for k in v["keys"]:
        scene = bytes.fromhex(k["scene_key"])
        assert dac1.key_id(scene).hex() == k["key_id"]
        # The client never derives scene keys, but its HKDF is the same one.
        info = f"{k['user_id']}\0{k['scene_id']}\0{k['epoch']}".encode()
        derived = dac1.hkdf_sha256(k["secret"].encode(), b"digiarena/render-cache/v1", info)
        assert derived == scene
    for f in v["files"]:
        scene = bytes.fromhex(v["keys"][f["key"]]["scene_key"])
        assert dac1.file_key(scene, f["digest"], f["chunk_size"]).hex() == f["file_key"]
        assert dac1.file_name(scene, f["digest"]) == f["file_name"]


@pytest.mark.parametrize("piece", [1, 7, 16, 4096, None], ids=lambda p: f"piece={p}")
def test_every_vector_decrypts_in_pieces_of_any_size(piece):
    v = _vectors()
    for f in v["files"]:
        scene = bytes.fromhex(v["keys"][f["key"]]["scene_key"])
        plain = ref.plaintext_of(f["plaintext"])
        if "ciphertext" in f:
            data = bytes.fromhex(f["ciphertext"])
        else:  # large vectors keep only their hash: this client's encryption must match
            data = dac1.encrypt(scene, f["digest"], plain, f["chunk_size"])
            assert hashlib.sha256(data).hexdigest() == f["sha256"]
        assert data[: dac1.HEADER_LEN].hex() == f["header"] and len(data) == f["length"]
        decryptor = dac1.Decryptor(scene, f["digest"])
        out = []
        step = piece or len(data)
        for at in range(0, len(data), step):
            out += decryptor.update(data[at : at + step])
        decryptor.finish()
        assert b"".join(out) == plain


def test_the_client_encrypts_like_the_reference():
    v = _vectors()
    for f in v["files"]:
        if "ciphertext" not in f:
            continue
        scene = bytes.fromhex(v["keys"][f["key"]]["scene_key"])
        plain = ref.plaintext_of(f["plaintext"])
        assert dac1.encrypt(scene, f["digest"], plain, f["chunk_size"]).hex() == f["ciphertext"]


def _small(v):
    f = v["files"][4]  # 33 bytes in 16-byte chunks: two full chunks and one byte
    return (
        bytes.fromhex(v["keys"][f["key"]]["scene_key"]),
        f["digest"],
        bytes.fromhex(f["ciphertext"]),
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "flip a ciphertext byte",
        "flip a tag byte",
        "drop the last chunk",
        "drop a byte",
        "append a byte",
        "swap two chunks",
        "change the length field",
        "change the key id",
        "set a reserved byte",
        "truncate and rewrite the header",
        "another file's header",
    ],
)
def test_a_tampered_file_does_not_decrypt(tamper):
    v = _vectors()
    k, digest, ct = _small(v)
    b = bytearray(ct)
    c = 16 + dac1.TAG_LEN
    if tamper == "flip a ciphertext byte":
        b[dac1.HEADER_LEN + 3] ^= 1
    elif tamper == "flip a tag byte":
        b[-1] ^= 0x80
    elif tamper == "drop the last chunk":
        del b[dac1.HEADER_LEN + 2 * c :]
    elif tamper == "drop a byte":
        del b[-1]
    elif tamper == "append a byte":
        b.append(0)
    elif tamper == "swap two chunks":
        first = b[dac1.HEADER_LEN : dac1.HEADER_LEN + c]
        b[dac1.HEADER_LEN : dac1.HEADER_LEN + c] = b[dac1.HEADER_LEN + c : dac1.HEADER_LEN + 2 * c]
        b[dac1.HEADER_LEN + c : dac1.HEADER_LEN + 2 * c] = first
    elif tamper == "change the length field":
        b[19] = 32
    elif tamper == "change the key id":
        b[30] ^= 1
    elif tamper == "set a reserved byte":
        b[63] = 1
    elif tamper == "truncate and rewrite the header":
        head = dac1.make_header(32, 16, dac1.key_id(k))
        b = bytearray(head + bytes(b[dac1.HEADER_LEN : dac1.HEADER_LEN + 2 * c]))
    elif tamper == "another file's header":
        other = v["files"][3]  # 32 bytes under the same key
        b[: dac1.HEADER_LEN] = bytes.fromhex(other["ciphertext"])[: dac1.HEADER_LEN]
    with pytest.raises(dac1.Dac1Error):
        dac1.decrypt(k, digest, bytes(b))
    # In pieces too, and a chunk's plaintext only comes out after its tag:
    # whatever was handed out before the failure is the real plaintext.
    plain = ref.plaintext_of({"pattern": "dac1/partial", "length": 33})
    decryptor, released = dac1.Decryptor(k, digest), []
    with pytest.raises(dac1.Dac1Error):
        for at in range(0, len(b), 5):
            released += decryptor.update(bytes(b[at : at + 5]))
        decryptor.finish()
    out = b"".join(released)
    assert out == plain[: len(out)]


def test_another_key_or_digest_does_not_decrypt():
    v = _vectors()
    k, digest, ct = _small(v)
    rotated = bytes.fromhex(v["keys"][1]["scene_key"])  # the next epoch
    with pytest.raises(dac1.Dac1Error, match="another key"):
        dac1.decrypt(rotated, digest, ct)
    with pytest.raises(dac1.Dac1Error):
        dac1.decrypt(k, hashlib.sha256(b"other").hexdigest(), ct)


@pytest.mark.parametrize("bad", ["", "A" * 64, "g" * 64, "a" * 63, "../" + "a" * 61, None])
def test_digests_must_be_64_lowercase_hex(bad):
    k = dac1.hkdf_sha256(b"s" * 32, b"", b"x")
    assert not dac1.is_digest(bad)
    with pytest.raises(dac1.Dac1Error):
        dac1.file_name(k, bad)
    assert dac1.is_digest("ab" * 32)


def test_header_limits():
    kid = bytes(16)
    for size in (dac1.MIN_CHUNK - 1, dac1.MAX_CHUNK + 1):
        head = bytearray(dac1.make_header(10, 16, kid))
        head[8:12] = size.to_bytes(4, "big")
        with pytest.raises(dac1.Dac1Error, match="chunk size"):
            dac1.parse_header(bytes(head))
    with pytest.raises(dac1.Dac1Error, match="version"):
        head = bytearray(dac1.make_header(10, 16, kid))
        head[4] = 2
        dac1.parse_header(bytes(head))
    with pytest.raises(dac1.Dac1Error, match="not a DAC1"):
        dac1.parse_header(b"ply\n" + bytes(60))
