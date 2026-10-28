"""DAC1 reference implementation: the encrypted scene asset format, in Python.

The platform gateway (digiarena-website ``src/lib/dac1.ts``, Node crypto)
encrypts scene assets on the way out; the browser editor and viewer decrypt
them with WebCrypto (``frontend/src/dac1.ts``); the Python client will store
them as they arrive. This module is the third, independent implementation:
AES-256 and GCM written out here (stdlib plus numpy), so the shared test
vectors in ``tests/data/dac1_vectors.json`` are checked against code that
shares nothing with OpenSSL or BoringSSL. It is a test oracle, not runtime
code: it is slow.

The format is specified in docs/dev/dac1_format.md. In short:

* ``K_scene = HKDF-SHA256(ikm=secret, salt="digiarena/render-cache/v1",
  info=user_id 0x00 scene_id 0x00 epoch)``, 32 bytes.
* ``key_id = HMAC-SHA256(K_scene, "key-id")[:16]``.
* ``K_file = HKDF-SHA256(ikm=K_scene, salt="", info="file" u8(version)
  u32be(chunk_size) digest)``, 32 bytes; ``digest`` is the 32-byte SHA-256
  of the plaintext.
* A 64-byte header, then one AES-256-GCM chunk per ``chunk_size`` bytes of
  plaintext (at least one chunk), each followed by its 16-byte tag. Nonce
  ``u64be(index) 0x00000000``; AAD ``SHA-256(header) u64be(index) u8(last)``.

Run ``python tests/dac1_ref.py --write`` to regenerate the vectors file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"DAC1"
VERSION = 1
HEADER_LEN = 64
TAG_LEN = 16
CHUNK_SIZE = 4 * 1024 * 1024
MIN_CHUNK = 16
MAX_CHUNK = 64 * 1024 * 1024
SCENE_SALT = b"digiarena/render-cache/v1"

VECTORS = Path(__file__).resolve().parent / "data" / "dac1_vectors.json"


class Dac1Error(ValueError):
    """The bytes are not a valid DAC1 file for this key and digest."""


# --------------------------------------------------------------------------- #
# HKDF (RFC 5869) and helpers
# --------------------------------------------------------------------------- #
def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    prk = hmac.new(salt or b"\0" * 32, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def scene_key(secret: str, user_id: str, scene_id: str, epoch: str) -> bytes:
    info = user_id.encode() + b"\0" + scene_id.encode() + b"\0" + epoch.encode()
    return hkdf_sha256(secret.encode(), SCENE_SALT, info)


def key_id(k_scene: bytes) -> bytes:
    return hmac.new(k_scene, b"key-id", hashlib.sha256).digest()[:16]


def _digest_bytes(digest_hex: str) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{64}", digest_hex):
        raise Dac1Error("digest must be 64 lowercase hex characters")
    return bytes.fromhex(digest_hex)


def file_key(k_scene: bytes, digest_hex: str, chunk_size: int, version: int = VERSION) -> bytes:
    info = b"file" + bytes([version]) + struct.pack(">I", chunk_size) + _digest_bytes(digest_hex)
    return hkdf_sha256(k_scene, b"", info)


def file_name(k_scene: bytes, digest_hex: str) -> str:
    return hmac.new(k_scene, b"name" + _digest_bytes(digest_hex), hashlib.sha256).hexdigest()[:32]


def chunk_count(length: int, chunk_size: int) -> int:
    return max(1, -(-length // chunk_size))


def cipher_length(length: int, chunk_size: int) -> int:
    return HEADER_LEN + length + TAG_LEN * chunk_count(length, chunk_size)


def header(length: int, chunk_size: int, kid: bytes) -> bytes:
    if len(kid) != 16:
        raise Dac1Error("key_id must be 16 bytes")
    out = (
        MAGIC
        + bytes([VERSION, 0, 0, 0])
        + struct.pack(">IQI", chunk_size, length, chunk_count(length, chunk_size))
        + kid
    )
    return out + b"\0" * (HEADER_LEN - len(out))


def parse_header(data: bytes) -> dict:
    if len(data) < HEADER_LEN:
        raise Dac1Error("shorter than a DAC1 header")
    h = data[:HEADER_LEN]
    if h[:4] != MAGIC:
        raise Dac1Error("not a DAC1 file")
    if h[4] != VERSION:
        raise Dac1Error(f"unsupported DAC1 version {h[4]}")
    if h[5:8] != b"\0\0\0" or h[40:64] != b"\0" * 24:
        raise Dac1Error("reserved header bytes are not zero")
    chunk_size, length, chunks = struct.unpack(">IQI", h[8:24])
    if not MIN_CHUNK <= chunk_size <= MAX_CHUNK:
        raise Dac1Error(f"chunk size {chunk_size} out of range")
    if chunks != chunk_count(length, chunk_size):
        raise Dac1Error("chunk count does not match the length")
    return {"chunk_size": chunk_size, "length": length, "chunks": chunks, "key_id": h[24:40]}


# --------------------------------------------------------------------------- #
# AES-256 (FIPS-197), table-driven
# --------------------------------------------------------------------------- #
def _xtime(a: int) -> int:
    a <<= 1
    return (a ^ 0x11B) if a & 0x100 else a


def _gmul(a: int, b: int) -> int:
    p = 0
    while b:
        if b & 1:
            p ^= a
        a = _xtime(a)
        b >>= 1
    return p


def _build_sbox() -> list[int]:
    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if _gmul(a, b) == 1:
                inv[a] = b
                break
    box = []
    for a in range(256):
        x = inv[a]
        y = x
        for _ in range(4):
            x = ((x << 1) | (x >> 7)) & 0xFF
            y ^= x
        box.append(y ^ 0x63)
    return box


SBOX = _build_sbox()


def _rotr(w: int, n: int) -> int:
    return ((w >> n) | (w << (32 - n))) & 0xFFFFFFFF


_T0 = [
    (_gmul(s, 2) << 24) | (s << 16) | (s << 8) | _gmul(s, 3) for s in SBOX
]  # MixColumns column for the first state byte
_T1 = [_rotr(t, 8) for t in _T0]
_T2 = [_rotr(t, 16) for t in _T0]
_T3 = [_rotr(t, 24) for t in _T0]
_NT = [np.array(t, dtype=np.uint32) for t in (_T0, _T1, _T2, _T3)]
_NSBOX = np.array(SBOX, dtype=np.uint32)


def expand_key(key: bytes) -> list[int]:
    if len(key) != 32:
        raise ValueError("AES-256 needs a 32-byte key")
    words = list(struct.unpack(">8I", key))
    rcon = 1
    for i in range(8, 60):
        t = words[i - 1]
        if i % 8 == 0:
            t = ((t << 8) | (t >> 24)) & 0xFFFFFFFF
            t = (SBOX[t >> 24] << 24) | (SBOX[(t >> 16) & 255] << 16) | (
                SBOX[(t >> 8) & 255] << 8
            ) | SBOX[t & 255]
            t ^= rcon << 24
            rcon = _xtime(rcon)
        elif i % 8 == 4:
            t = (SBOX[t >> 24] << 24) | (SBOX[(t >> 16) & 255] << 16) | (
                SBOX[(t >> 8) & 255] << 8
            ) | SBOX[t & 255]
        words.append(words[i - 8] ^ t)
    return words


def encrypt_block(rk: list[int], block: bytes) -> bytes:
    s0, s1, s2, s3 = struct.unpack(">4I", block)
    s0 ^= rk[0]
    s1 ^= rk[1]
    s2 ^= rk[2]
    s3 ^= rk[3]
    T0, T1, T2, T3 = _T0, _T1, _T2, _T3
    for r in range(1, 14):
        k = 4 * r
        t0 = T0[s0 >> 24] ^ T1[(s1 >> 16) & 255] ^ T2[(s2 >> 8) & 255] ^ T3[s3 & 255] ^ rk[k]
        t1 = T0[s1 >> 24] ^ T1[(s2 >> 16) & 255] ^ T2[(s3 >> 8) & 255] ^ T3[s0 & 255] ^ rk[k + 1]
        t2 = T0[s2 >> 24] ^ T1[(s3 >> 16) & 255] ^ T2[(s0 >> 8) & 255] ^ T3[s1 & 255] ^ rk[k + 2]
        t3 = T0[s3 >> 24] ^ T1[(s0 >> 16) & 255] ^ T2[(s1 >> 8) & 255] ^ T3[s2 & 255] ^ rk[k + 3]
        s0, s1, s2, s3 = t0, t1, t2, t3

    def last(a, b, c, d, w):
        return (
            (SBOX[a >> 24] << 24) | (SBOX[(b >> 16) & 255] << 16) | (SBOX[(c >> 8) & 255] << 8)
            | SBOX[d & 255]
        ) ^ w

    return struct.pack(
        ">4I",
        last(s0, s1, s2, s3, rk[56]),
        last(s1, s2, s3, s0, rk[57]),
        last(s2, s3, s0, s1, rk[58]),
        last(s3, s0, s1, s2, rk[59]),
    )


def _encrypt_blocks(rk: list[int], words: np.ndarray) -> np.ndarray:
    """AES-256 of many blocks at once: ``words`` is (n, 4) uint32 big-endian
    words, one row per block. The same rounds as :func:`encrypt_block`."""
    s = [words[:, j] ^ np.uint32(rk[j]) for j in range(4)]
    T0, T1, T2, T3 = _NT
    for r in range(1, 14):
        k = 4 * r
        s = [
            T0[s[j] >> 24]
            ^ T1[(s[(j + 1) % 4] >> 16) & 255]
            ^ T2[(s[(j + 2) % 4] >> 8) & 255]
            ^ T3[s[(j + 3) % 4] & 255]
            ^ np.uint32(rk[k + j])
            for j in range(4)
        ]
    S = _NSBOX
    out = [
        (
            (S[s[j] >> 24] << 24)
            | (S[(s[(j + 1) % 4] >> 16) & 255] << 16)
            | (S[(s[(j + 2) % 4] >> 8) & 255] << 8)
            | S[s[(j + 3) % 4] & 255]
        )
        ^ np.uint32(rk[56 + j])
        for j in range(4)
    ]
    return np.stack(out, axis=1).astype(np.uint32)


# --------------------------------------------------------------------------- #
# GCM (NIST SP 800-38D) with a 96-bit IV
# --------------------------------------------------------------------------- #
_R = 0xE1 << 120


def _gf_mult(x: int, y: int) -> int:
    z, v = 0, y
    for i in range(127, -1, -1):
        if (x >> i) & 1:
            z ^= v
        v = (v >> 1) ^ _R if v & 1 else v >> 1
    return z


class _Ghash:
    """Multiplication by a fixed H through per-byte tables (16 x 256)."""

    def __init__(self, h: int):
        self.tables = []
        for pos in range(16):
            shift = 8 * (15 - pos)
            self.tables.append([_gf_mult(b << shift, h) for b in range(256)])

    def mult(self, x: int) -> int:
        z = 0
        for pos, table in enumerate(self.tables):
            z ^= table[(x >> (8 * (15 - pos))) & 255]
        return z

    def digest(self, aad: bytes, ct: bytes) -> int:
        y = 0
        for data in (aad, ct):
            padded = data + b"\0" * (-len(data) % 16)
            for i in range(0, len(padded), 16):
                y = self.mult(y ^ int.from_bytes(padded[i : i + 16], "big"))
        y ^= ((len(aad) * 8) << 64) | (len(ct) * 8)
        return self.mult(y)


class AesGcm:
    def __init__(self, key: bytes):
        self.rk = expand_key(key)
        self.ghash = _Ghash(int.from_bytes(encrypt_block(self.rk, b"\0" * 16), "big"))

    def _ctr(self, iv: bytes, data: bytes) -> bytes:
        n = -(-len(data) // 16)
        if n == 0:
            return b""
        counters = np.empty((n, 4), dtype=np.uint32)
        counters[:, :3] = np.frombuffer(iv, dtype=">u4").astype(np.uint32)
        # inc32 of J0 = IV || 1: the first data block uses counter 2.
        counters[:, 3] = (np.arange(n, dtype=np.uint64) + 2).astype(np.uint32)
        stream = _encrypt_blocks(self.rk, counters).astype(">u4").tobytes()[: len(data)]
        out = np.frombuffer(data, dtype=np.uint8) ^ np.frombuffer(stream, dtype=np.uint8)
        return out.tobytes()

    def _tag(self, iv: bytes, aad: bytes, ct: bytes) -> bytes:
        s = self.ghash.digest(aad, ct)
        ek0 = int.from_bytes(encrypt_block(self.rk, iv + b"\0\0\0\1"), "big")
        return (s ^ ek0).to_bytes(16, "big")

    def encrypt(self, iv: bytes, plaintext: bytes, aad: bytes) -> bytes:
        if len(iv) != 12:
            raise ValueError("GCM IV must be 12 bytes here")
        ct = self._ctr(iv, plaintext)
        return ct + self._tag(iv, aad, ct)

    def decrypt(self, iv: bytes, data: bytes, aad: bytes) -> bytes:
        if len(data) < 16:
            raise Dac1Error("chunk shorter than its tag")
        ct, tag = data[:-16], data[-16:]
        if not hmac.compare_digest(self._tag(iv, aad, ct), tag):
            raise Dac1Error("authentication failed")
        return self._ctr(iv, ct)


# --------------------------------------------------------------------------- #
# DAC1
# --------------------------------------------------------------------------- #
def _nonce(index: int) -> bytes:
    return struct.pack(">QI", index, 0)


def _aad(header_hash: bytes, index: int, last: bool) -> bytes:
    return header_hash + struct.pack(">QB", index, 1 if last else 0)


def encrypt(
    k_scene: bytes, digest_hex: str, plaintext: bytes, chunk_size: int = CHUNK_SIZE
) -> bytes:
    if hashlib.sha256(plaintext).hexdigest() != digest_hex:
        raise Dac1Error("plaintext does not match its digest")
    head = header(len(plaintext), chunk_size, key_id(k_scene))
    head_hash = hashlib.sha256(head).digest()
    gcm = AesGcm(file_key(k_scene, digest_hex, chunk_size))
    n = chunk_count(len(plaintext), chunk_size)
    parts = [head]
    for i in range(n):
        chunk = plaintext[i * chunk_size : (i + 1) * chunk_size]
        parts.append(gcm.encrypt(_nonce(i), chunk, _aad(head_hash, i, i == n - 1)))
    return b"".join(parts)


def decrypt(k_scene: bytes, digest_hex: str, data: bytes) -> bytes:
    meta = parse_header(data)
    if meta["key_id"] != key_id(k_scene):
        raise Dac1Error("encrypted for another key")
    chunk_size, length, n = meta["chunk_size"], meta["length"], meta["chunks"]
    if len(data) != cipher_length(length, chunk_size):
        raise Dac1Error("truncated or padded")
    head_hash = hashlib.sha256(data[:HEADER_LEN]).digest()
    gcm = AesGcm(file_key(k_scene, digest_hex, chunk_size))
    out, at = [], HEADER_LEN
    for i in range(n):
        size = min(chunk_size, length - i * chunk_size) + TAG_LEN
        out.append(gcm.decrypt(_nonce(i), data[at : at + size], _aad(head_hash, i, i == n - 1)))
        at += size
    plain = b"".join(out)
    if hashlib.sha256(plain).hexdigest() != digest_hex:
        raise Dac1Error("plaintext does not match its digest")
    return plain


def pattern(seed: str, length: int) -> bytes:
    """Deterministic test bytes: SHA-256(seed || u32be(i)) for i = 0, 1, ..."""
    out = bytearray()
    i = 0
    while len(out) < length:
        out += hashlib.sha256(seed.encode() + struct.pack(">I", i)).digest()
        i += 1
    return bytes(out[:length])


# --------------------------------------------------------------------------- #
# The shared vectors
# --------------------------------------------------------------------------- #
KEYS = [
    {
        "secret": "test-only-render-cache-secret-0000000000000000",
        "user_id": "8d3c2f0e-1b7a-4c55-9e0d-2a6f4b1c9e77",
        "scene_id": "5f2b7d10-6c3e-4a8b-b1d9-0e4c7a2f6b31",
        "epoch": "1.0",
    },
    {
        "secret": "test-only-render-cache-secret-0000000000000000",
        "user_id": "8d3c2f0e-1b7a-4c55-9e0d-2a6f4b1c9e77",
        "scene_id": "5f2b7d10-6c3e-4a8b-b1d9-0e4c7a2f6b31",
        "epoch": "1.1",
    },
    {
        "secret": "another secret, at least thirty-two bytes long",
        "user_id": "0b9e4d2a-7f61-4e3c-8a15-c2d0f9b7e468",
        "scene_id": "5f2b7d10-6c3e-4a8b-b1d9-0e4c7a2f6b31",
        "epoch": "7.0",
    },
]

# (key index, plaintext spec, chunk size, keep the full ciphertext in the file)
FILES = [
    (0, {"text": ""}, CHUNK_SIZE, True),
    (0, {"text": "p"}, CHUNK_SIZE, True),
    (0, {"text": '{"scene_id": "dac1", "objects": []}\n'}, CHUNK_SIZE, True),
    (0, {"pattern": "dac1/exact", "length": 32}, 16, True),
    (0, {"pattern": "dac1/partial", "length": 33}, 16, True),
    (1, {"pattern": "dac1/partial", "length": 33}, 16, True),
    (2, {"pattern": "dac1/partial", "length": 33}, 16, True),
    (0, {"pattern": "dac1/odd", "length": 1000}, 64, True),
    (2, {"pattern": "dac1/medium", "length": 200 * 1024}, 64 * 1024, False),
    (0, {"pattern": "dac1/production", "length": CHUNK_SIZE + 7}, CHUNK_SIZE, False),
]


def plaintext_of(spec: dict) -> bytes:
    if "text" in spec:
        return spec["text"].encode()
    return pattern(spec["pattern"], spec["length"])


def build_vectors() -> dict:
    keys = []
    for k in KEYS:
        ks = scene_key(k["secret"], k["user_id"], k["scene_id"], k["epoch"])
        keys.append({**k, "scene_key": ks.hex(), "key_id": key_id(ks).hex()})
    files = []
    for index, spec, chunk_size, full in FILES:
        ks = bytes.fromhex(keys[index]["scene_key"])
        plain = plaintext_of(spec)
        digest = hashlib.sha256(plain).hexdigest()
        ct = encrypt(ks, digest, plain, chunk_size)
        entry = {
            "key": index,
            "plaintext": spec,
            "digest": digest,
            "chunk_size": chunk_size,
            "file_key": file_key(ks, digest, chunk_size).hex(),
            "file_name": file_name(ks, digest),
            "header": ct[:HEADER_LEN].hex(),
            "length": len(ct),
            "sha256": hashlib.sha256(ct).hexdigest(),
        }
        if full:
            entry["ciphertext"] = ct.hex()
        files.append(entry)
    return {
        "format": "DAC1",
        "version": VERSION,
        "spec": "world-engine docs/dev/dac1_format.md",
        "keys": keys,
        "files": files,
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        VECTORS.parent.mkdir(parents=True, exist_ok=True)
        VECTORS.write_text(json.dumps(build_vectors(), indent=1) + "\n")
        print(f"wrote {VECTORS}")
    else:
        print("usage: python tests/dac1_ref.py --write")
