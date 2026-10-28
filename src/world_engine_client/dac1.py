"""DAC1: the encrypted form of a scene asset, on the wire and on this disk.

The hosting platform's gateway encrypts every scene asset it sends; this client
stores the bytes exactly as they arrive and decrypts them in memory only. The
format is specified in the engine repository (``docs/dev/dac1_format.md``). In
short, with ``K_scene`` the 32-byte key the platform issues for one user and
one scene:

* ``key_id = HMAC-SHA256(K_scene, "key-id")[:16]``;
* ``K_file = HKDF-SHA256(ikm=K_scene, salt="", info="file" u8(version)
  u32be(chunk_size) digest)``, ``digest`` being the 32-byte SHA-256 of the
  plaintext;
* the file name a client stores it under is
  ``hex(HMAC-SHA256(K_scene, "name" digest))[:32]``, so a disk does not show
  which public digest it holds;
* a 64-byte header (magic, version, chunk size, plaintext length, chunk count,
  key id), then one AES-256-GCM chunk per ``chunk_size`` bytes of plaintext,
  each followed by its 16-byte tag. Nonce ``u64be(index) 0x00000000``; AAD
  ``SHA-256(header) u64be(index) u8(last)``.

Key derivation needs only the standard library. AES-GCM needs the
``cryptography`` package (the ``render`` extra), imported when first used.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"DAC1"
VERSION = 1
HEADER_LEN = 64
TAG_LEN = 16
CHUNK_SIZE = 4 * 1024 * 1024
MIN_CHUNK = 16
MAX_CHUNK = 64 * 1024 * 1024
KEY_LEN = 32

_DIGEST = re.compile(r"[0-9a-f]{64}")


class Dac1Error(ValueError):
    """The bytes are not a valid DAC1 file for this key and digest."""


def is_digest(value: object) -> bool:
    """True for a SHA-256 digest in the form the platform sends: 64 lowercase
    hexadecimal characters."""
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _digest_bytes(digest: str) -> bytes:
    if not is_digest(digest):
        raise Dac1Error("digest must be 64 lowercase hex characters")
    return bytes.fromhex(digest)


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = KEY_LEN) -> bytes:
    """HKDF (RFC 5869) with SHA-256. An empty salt means 32 zero bytes."""
    prk = hmac.new(salt or b"\0" * 32, ikm, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def key_id(scene_key: bytes) -> bytes:
    """The key's public name (16 bytes): it changes whenever the key does."""
    return hmac.new(bytes(scene_key), b"key-id", hashlib.sha256).digest()[:16]


def file_key(scene_key: bytes, digest: str, chunk_size: int, version: int = VERSION) -> bytes:
    """The AES-256 key of one file: one per digest, chunk size and version."""
    info = b"file" + bytes([version]) + struct.pack(">I", chunk_size) + _digest_bytes(digest)
    return hkdf_sha256(bytes(scene_key), b"", info)


def file_name(scene_key: bytes, digest: str) -> str:
    """The name this client stores the file under (32 hex characters)."""
    mac = hmac.new(bytes(scene_key), b"name" + _digest_bytes(digest), hashlib.sha256)
    return mac.hexdigest()[:32]


def chunk_count(length: int, chunk_size: int) -> int:
    return max(1, -(-length // chunk_size))


def cipher_length(length: int, chunk_size: int) -> int:
    """The size of a DAC1 file whose plaintext is ``length`` bytes."""
    return HEADER_LEN + length + TAG_LEN * chunk_count(length, chunk_size)


@dataclass(frozen=True)
class Header:
    chunk_size: int
    length: int  # of the plaintext
    chunks: int
    key_id: bytes
    raw: bytes

    @property
    def cipher_length(self) -> int:
        return cipher_length(self.length, self.chunk_size)

    def sealed_size(self, index: int) -> int:
        """Bytes of chunk ``index`` on disk: its ciphertext and its tag."""
        start = index * self.chunk_size
        return min(self.chunk_size, self.length - start) + TAG_LEN


def make_header(length: int, chunk_size: int, kid: bytes) -> bytes:
    if len(kid) != 16:
        raise Dac1Error("key_id must be 16 bytes")
    body = MAGIC + bytes([VERSION, 0, 0, 0])
    body += struct.pack(">IQI", chunk_size, length, chunk_count(length, chunk_size)) + kid
    return body + b"\0" * (HEADER_LEN - len(body))


def parse_header(data: bytes) -> Header:
    if len(data) < HEADER_LEN:
        raise Dac1Error("shorter than a DAC1 header")
    h = bytes(data[:HEADER_LEN])
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
    return Header(chunk_size, length, chunks, h[24:40], h)


def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # pragma: no cover - the render extra brings it
        raise Dac1Error(
            "decrypting scene assets needs the cryptography package: "
            "pip install 'world-engine-client[render]'"
        ) from exc
    return AESGCM(key)


def _nonce(index: int) -> bytes:
    return struct.pack(">QI", index, 0)


def _aad(header_hash: bytes, index: int, last: bool) -> bytes:
    return header_hash + struct.pack(">QB", index, 1 if last else 0)


class Decryptor:
    """Decrypt a DAC1 file fed in pieces of any size, checking everything.

    ``update(data)`` returns the plaintext of each chunk the data completed,
    after its tag verified; ``finish()`` checks that the file ended where its
    header says and that the plaintext's SHA-256 is ``digest``. Every failure
    raises :class:`Dac1Error`: a wrong key, a changed byte, a missing or extra
    byte, chunks out of order, another file's header."""

    def __init__(self, scene_key: bytes, digest: str):
        self._scene_key = bytes(scene_key)
        self._digest = digest
        _digest_bytes(digest)
        self._key_id = key_id(self._scene_key)
        self._buf = bytearray()
        self._header: Header | None = None
        self._gcm = None
        self._header_hash = b""
        self._index = 0
        self._sha = hashlib.sha256()
        self.received = 0  # bytes fed so far

    @property
    def header(self) -> Header | None:
        return self._header

    @property
    def done(self) -> bool:
        return self._header is not None and self._index == self._header.chunks

    def update(self, data: bytes) -> list[bytes]:
        self.received += len(data)
        self._buf += data
        out: list[bytes] = []
        if self._header is None:
            if len(self._buf) < HEADER_LEN:
                return out
            header = parse_header(self._buf)
            if not hmac.compare_digest(header.key_id, self._key_id):
                raise Dac1Error("encrypted for another key")
            self._header = header
            self._header_hash = hashlib.sha256(header.raw).digest()
            self._gcm = _aesgcm(file_key(self._scene_key, self._digest, header.chunk_size))
            del self._buf[:HEADER_LEN]
        header = self._header
        while self._index < header.chunks:
            size = header.sealed_size(self._index)
            if len(self._buf) < size:
                break
            sealed = bytes(self._buf[:size])
            del self._buf[:size]
            last = self._index == header.chunks - 1
            try:
                plain = self._gcm.decrypt(
                    _nonce(self._index), sealed, _aad(self._header_hash, self._index, last)
                )
            except Exception as exc:  # cryptography's InvalidTag
                raise Dac1Error(f"chunk {self._index} failed authentication") from exc
            self._sha.update(plain)
            out.append(plain)
            self._index += 1
        if self._index == header.chunks and self._buf:
            raise Dac1Error("bytes after the last chunk")
        return out

    def finish(self) -> None:
        if self._header is None or self._index != self._header.chunks:
            raise Dac1Error("truncated")
        if self._buf:
            raise Dac1Error("bytes after the last chunk")
        if self._sha.hexdigest() != self._digest:
            raise Dac1Error("plaintext does not match its digest")


def encrypt(scene_key: bytes, digest: str, plaintext: bytes, chunk_size: int = CHUNK_SIZE) -> bytes:
    """Encrypt ``plaintext`` the way the platform's gateway does (tests, tools)."""
    if hashlib.sha256(plaintext).hexdigest() != digest:
        raise Dac1Error("plaintext does not match its digest")
    head = make_header(len(plaintext), chunk_size, key_id(scene_key))
    head_hash = hashlib.sha256(head).digest()
    gcm = _aesgcm(file_key(scene_key, digest, chunk_size))
    n = chunk_count(len(plaintext), chunk_size)
    parts = [head]
    for i in range(n):
        chunk = plaintext[i * chunk_size : (i + 1) * chunk_size]
        parts.append(gcm.encrypt(_nonce(i), chunk, _aad(head_hash, i, i == n - 1)))
    return b"".join(parts)


def decrypt(scene_key: bytes, digest: str, data: bytes) -> bytes:
    """The plaintext of a whole DAC1 file in memory, checked."""
    decryptor = Decryptor(scene_key, digest)
    out = decryptor.update(data)
    decryptor.finish()
    return b"".join(out)


def read_header(path: str | Path) -> Header:
    with open(path, "rb") as f:
        return parse_header(f.read(HEADER_LEN))


def iter_file(
    scene_key: bytes, digest: str, path: str | Path, piece: int = 1 << 20
) -> Iterator[bytes]:
    """The verified plaintext of a DAC1 file on disk, chunk by chunk. Each
    chunk is yielded only after its tag verified; the generator raises
    :class:`Dac1Error` at the first problem, at the latest after the last
    chunk (a wrong size or digest)."""
    decryptor = Decryptor(scene_key, digest)
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(piece), b""):
            yield from decryptor.update(data)
    decryptor.finish()
