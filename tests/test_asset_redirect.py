"""Scene assets straight from the platform's storage.

Where the platform offers it, a request for a scene asset that carries
``X-We-Render-Redirect: 1`` gets a 302 to a presigned URL of the file's
ciphertext in the platform's storage, valid for minutes; a request without
the header gets the DAC1 stream as before. Here a stand-in for the platform
(platform_stub.py on a ControlStub) and one for its storage run in this
process. The client downloads the scene through SceneAssets with the env's own
fetch, as a renderer does, and the tests look at what each server received and
what the cache holds.
"""

from __future__ import annotations

import datetime
import hashlib
import ipaddress
import logging
import socket
import ssl
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import dac1_ref as ref
import pytest
from conftest import send_bytes, start_stub
from platform_stub import ASSETS, ROBOT, SCENE, Platform, stream

import world_engine_client as wec
from world_engine_client import env as env_module
from world_engine_client.errors import RenderFetchError

pytest.importorskip("cryptography")

URLS = sorted(ASSETS)  # the order SceneAssets downloads them in
DIGESTS = {url: hashlib.sha256(data).hexdigest() for url, data in {**ASSETS, **ROBOT}.items()}
# What http.client sends to storage by itself.
OWN = {"Host", "Accept-Encoding"}
CREDENTIALS = ("X-We-Control-Token", "Authorization", "Cookie", "X-We-Render-Redirect")


# ----- the stand-ins --------------------------------------------------------- #
class Storage:
    """The platform's storage on a loopback port: GET serves the object at the
    URL's path, with byte ranges, whatever the query says. ``requests`` has
    the path, signature, headers and status of each request. ``answers`` is
    what the next requests get instead of the object: a status, ``("cut", n)``
    for a body that breaks off after n bytes, ``("bytes", data)`` or
    ``("redirect", url)``."""

    def __init__(self, context=None):
        self.objects: dict[str, bytes] = {}
        self.requests: list[SimpleNamespace] = []
        self.answers: list = []
        self.lock = threading.Lock()
        storage = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # noqa: N802 - http.server API
                parts = urlsplit(self.path)
                query = parse_qs(parts.query)
                with storage.lock:
                    answer = storage.answers.pop(0) if storage.answers else None
                    seen = SimpleNamespace(
                        path=parts.path,
                        signature=query.get("X-Amz-Signature", [None])[0],
                        headers=dict(self.headers),
                        status=None,
                    )
                    storage.requests.append(seen)
                data = storage.objects.get(parts.path)
                if isinstance(answer, int):
                    seen.status = answer
                    body = b"<Error><Code>AccessDenied</Code></Error>"
                    self.send_response(answer)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif isinstance(answer, tuple) and answer[0] == "redirect":
                    seen.status = 302
                    self.send_response(302)
                    self.send_header("Location", answer[1])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif isinstance(answer, tuple) and answer[0] == "cut":
                    seen.status = "cut"
                    stream(self, data, {}, delay=0, cut=answer[1])
                elif data is None:
                    seen.status = 404
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    if isinstance(answer, tuple) and answer[0] == "bytes":
                        data = answer[1]
                    seen.status = 206 if self.headers.get("Range") else 200
                    send_bytes(self, data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        if context is not None:
            self.server.socket = context.wrap_socket(
                self.server.socket, server_side=True, do_handshake_on_connect=False
            )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        scheme = "https" if context is not None else "http"
        self.url = f"{scheme}://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Direct:
    """The platform's direct delivery on top of platform_stub.Platform, as
    the platform does it: a request for a scene asset that carries the header
    gets a 302 to a new presigned URL (a new signature each time) of the
    file's ciphertext in storage; a request without it streams.
    ``base`` is where the URLs point, ``where[url]`` another place for one
    asset (None: a redirect without a Location); the next ``streams[url]``
    requests of an asset stream even with
    the header (storage did not hold the file yet). ``also`` lists other
    paths that get a redirect whatever the request says."""

    def __init__(self, stub, platform: Platform, storage: Storage):
        self.platform = platform
        self.base = storage.url
        self.where: dict[str, str] = {}
        self.streams: dict[str, int] = {}
        self.signed: list[str] = []
        self.lock = threading.Lock()
        for url in platform.assets:
            if url.startswith("/__asset/"):
                storage.objects[self.object(url)] = platform.cipher(url)
                stub.gets["/render" + url] = self._redirect(url, stub.gets["/render" + url])
        self.also = set()
        for path in ("/render/__key", "/render/index.html", "/render/__robot/0.bin"):
            stub.gets[path] = self._other(path, stub.gets.get(path))

    def object(self, url: str) -> str:
        digest = self.platform.digests()[url]
        return f"/enc/1/{SCENE}/0/{ref.file_name(self.platform.key(), digest)}"

    def location(self, url: str) -> str | None:
        base = self.where.get(url, self.base)
        with self.lock:
            signature = f"{len(self.signed):064x}"
            self.signed.append(signature)
        query = f"X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature={signature}"
        return None if base is None else f"{base}{self.object(url)}?{query}"

    def _redirect(self, url: str, streamed):
        def answer(handler):
            with self.lock:
                stream_it = self.streams.get(url, 0) > 0
                if stream_it:
                    self.streams[url] -= 1
            if handler.headers.get("X-We-Render-Redirect") != "1" or stream_it:
                streamed(handler)
                return
            send_redirect(handler, self.location(url))

        return answer

    def _other(self, path: str, served):
        def answer(handler):
            if path in self.also:
                send_redirect(handler, f"{self.base}/elsewhere?X-Amz-Signature=other")
            elif served is not None:
                served(handler)
            else:
                body = self.platform.stub.files.get(path, b"")
                send_bytes(handler, body)

        return answer


def send_redirect(handler, location: str | None) -> None:
    handler.send_response(302)
    if location is not None:
        handler.send_header("Location", location)
    handler.send_header("Cache-Control", "no-store")
    # The platform's own cookie: it must not reach storage.
    handler.send_header("Set-Cookie", "session=platform-only; Path=/; HttpOnly")
    handler.send_header("Content-Length", "0")
    handler.end_headers()


@pytest.fixture
def storage():
    s = Storage()
    yield s
    s.close()


@pytest.fixture
def direct(stub, storage):
    return Direct(stub, Platform(stub), storage)


def connect(stub):
    """A platform session without images: only its fetch is used here."""
    return wec.connect("secret", url=stub.url, access_provider=lambda: "scoped", timeout=5)


def download(env, cache_dir, urls=None):
    """The scene's assets as a renderer gets them (SceneAssets with the env's
    fetch); returns the SceneAssets."""
    from world_engine_client.cache import SceneAssets

    assets = SceneAssets(
        env._open_render,
        {u: DIGESTS[u] for u in (urls or DIGESTS)},
        encrypted=True,
        platform=True,
        cache_dir=cache_dir,
        scene_id="fixture",
    )
    assets.prepare(cleanup=False)
    return assets


def asks(stub, url: str) -> list[tuple[bool, str | None]]:
    """(asked for a redirect, Range) of each request of a scene asset."""
    return [
        (h.get("X-We-Render-Redirect") == "1", h.get("Range"))
        for h in stub.get_calls("/render" + url)
    ]


def blobs(cache_dir) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in Path(cache_dir, "blobs").rglob("*") if p.is_file()}


def switch_lines(caplog) -> list[str]:
    """The env's line that it streams from now on."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "world_engine_client.env" and "from now on" in r.getMessage()
    ]


def resumes(caplog) -> list[str]:
    """SceneAssets' retries of a download that broke off ("...: resuming")."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "world_engine_client.cache" and "resuming" in r.getMessage()
    ]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def resolve(monkeypatch, names: dict[str, list[tuple[str, int]]]) -> list[tuple[str, int]]:
    """socket.getaddrinfo with each of ``names`` resolved to its loopback
    addresses, in that order, whatever the port; other names as before.
    Returns the (host, port) of each lookup of one of them."""
    real = socket.getaddrinfo
    asked = []

    def getaddrinfo(host, port, *args, **kwargs):
        if host not in names:
            return real(host, port, *args, **kwargs)
        asked.append((host, port))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", a) for a in names[host]]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return asked


# ----- following the redirect ----------------------------------------------- #
def test_a_redirect_is_followed_with_the_range_and_nothing_else(stub, storage, direct):
    env = connect(stub)
    url = "/__asset/0.ply"
    cipher = direct.platform.cipher(url)
    try:
        with env._open_render(url, {"Range": "bytes=100-"}) as response:
            assert response.status == 206
            assert response.getheader("Content-Range") == (
                f"bytes 100-{len(cipher) - 1}/{len(cipher)}"
            )
            assert response.read() == cipher[100:]
    finally:
        env.close()
    [platform] = stub.get_calls("/render" + url)
    assert platform["X-We-Render-Redirect"] == "1" and platform["Range"] == "bytes=100-"
    assert platform["X-We-Control-Token"] == "secret"
    assert platform["Authorization"] == "Bearer scoped"
    [got] = storage.requests
    assert got.path == direct.object(url) and got.signature == direct.signed[0]
    assert set(got.headers) == OWN | {"Range"}
    assert got.headers["Range"] == "bytes=100-"
    assert got.headers["Host"] == urlsplit(storage.url).netloc


def test_storage_gets_no_credentials_and_the_cache_is_as_if_streamed(
    stub, storage, direct, tmp_path
):
    """Every scene asset of the session from storage: the requests there
    carry neither the control token, nor the Authorization header, nor the
    platform's cookie, nor the header that asks for redirects. The cache
    ends up with the same files, under the same names, as from the stream."""
    env = connect(stub)
    try:
        assets = download(env, tmp_path / "direct")
        for url in URLS:
            assert assets.read(url) == ASSETS[url]
        assert [r.path for r in storage.requests] == [direct.object(u) for u in URLS]
        assert all(set(r.headers) <= OWN for r in storage.requests), storage.requests
        for r in storage.requests:
            assert not any(name in r.headers for name in CREDENTIALS)
        assert len(set(direct.signed)) == len(URLS)
        # The robot mesh is not a scene asset: it streams, unasked.
        assert stub.get_calls("/render/__robot/0.bin")[0].get("X-We-Render-Redirect") is None
        # A second session finds everything in the cache: it asks for the key
        # and nothing else, from neither server.
        before = len(stub.calls)
        download(env, tmp_path / "direct")
        assert [c[0] for c in stub.calls[before:]] == ["/render/__key"]
        assert len(storage.requests) == len(URLS)
    finally:
        env.close()
    streamed = start_stub()
    try:
        Platform(streamed)
        env = connect(streamed)
        try:
            download(env, tmp_path / "streamed")
        finally:
            env.close()
    finally:
        streamed.close()
    assert blobs(tmp_path / "direct") == blobs(tmp_path / "streamed")
    for url in URLS:
        assert direct.platform.cipher(url) in blobs(tmp_path / "direct").values()


def test_an_expired_url_gets_a_new_one_and_the_download_resumes(
    stub, storage, direct, tmp_path, caplog
):
    """The first part of the splat streams (storage did not hold it yet) and
    breaks off; the resumed request is redirected, and the URL is refused as
    expired. The client asks the platform again at once, gets a new URL and
    resumes with its Range."""
    caplog.set_level(logging.INFO, logger="world_engine_client")
    url = "/__asset/0.ply"
    direct.streams[url] = 1
    direct.platform.cut[url] = 10_000
    storage.answers = [403]
    env = connect(stub)
    try:
        assets = download(env, tmp_path, [url, "/__asset/1.glb"])
        assert assets.read(url) == ASSETS[url]
    finally:
        env.close()
    assert asks(stub, url) == [(True, None), (True, "bytes=10000-"), (True, "bytes=10000-")]
    first, second, other = storage.requests
    assert (first.status, second.status) == (403, 206)
    assert first.headers["Range"] == second.headers["Range"] == "bytes=10000-"
    assert first.signature != second.signature, "a URL that failed is used again"
    # One refused URL is forgiven: the next asset still comes from storage.
    assert asks(stub, "/__asset/1.glb") == [(True, None)]
    assert other.path == direct.object("/__asset/1.glb") and other.status == 200
    assert direct.platform.cipher(url) in blobs(tmp_path).values()
    assert len(resumes(caplog)) == 1, "the refused URL cost the download a retry"
    assert switch_lines(caplog) == []


def test_a_body_that_breaks_off_resumes_through_the_platform(
    stub, storage, direct, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger="world_engine_client")
    url = "/__asset/0.ply"
    storage.answers = [("cut", 10_000)]
    env = connect(stub)
    try:
        assets = download(env, tmp_path, [url, "/__asset/1.glb"])
        assert assets.read(url) == ASSETS[url]
    finally:
        env.close()
    assert asks(stub, url) == [(True, None), (True, "bytes=10000-")]
    first, second, other = storage.requests
    assert first.status == "cut" and "Range" not in first.headers
    assert second.status == 206 and second.headers["Range"] == "bytes=10000-"
    assert set(second.headers) == OWN | {"Range"}
    assert first.signature != second.signature
    assert asks(stub, "/__asset/1.glb") == [(True, None)]
    assert other.status == 200
    assert direct.platform.cipher(url) in blobs(tmp_path).values()
    assert len(resumes(caplog)) == 1 and switch_lines(caplog) == []


def test_a_second_failure_streams_for_the_rest_of_the_session(
    stub, storage, direct, tmp_path, caplog
):
    """A body that breaks off is one failure, a refused URL the second: the
    download resumes from the platform's stream, and so do the files after
    it, without asking for a redirect again. One log line, no warning."""
    caplog.set_level(logging.INFO, logger="world_engine_client")
    url = "/__asset/0.ply"
    storage.answers = [("cut", 10_000), 403]
    env = connect(stub)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assets = download(env, tmp_path)
        assert [str(w.message) for w in caught] == []
        for u in URLS:
            assert assets.read(u) == ASSETS[u]
    finally:
        env.close()
    assert asks(stub, url) == [(True, None), (True, "bytes=10000-"), (False, "bytes=10000-")]
    assert [r.status for r in storage.requests] == ["cut", 403]
    for u in URLS[1:]:
        assert asks(stub, u) == [(False, None)]
    [line] = switch_lines(caplog)
    assert "answered HTTP 403" in line and "second failure" in line
    assert len(resumes(caplog)) == 1
    assert "X-Amz-Signature" not in caplog.text and direct.signed[1] not in caplog.text


def test_storage_out_of_reach_streams_from_then_on(stub, storage, direct, tmp_path, caplog):
    """A port where nothing listens, as from a network that blocks storage:
    the first file streams from the platform at once, without a retry of the
    download, and every file after it is asked for as a stream."""
    caplog.set_level(logging.INFO, logger="world_engine_client")
    direct.base = f"http://127.0.0.1:{free_port()}"
    env = connect(stub)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assets = download(env, tmp_path)
        assert [str(w.message) for w in caught] == []
        for u in URLS:
            assert assets.read(u) == ASSETS[u]
    finally:
        env.close()
    assert asks(stub, URLS[0]) == [(True, None), (False, None)]
    for u in URLS[1:]:
        assert asks(stub, u) == [(False, None)]
    assert storage.requests == [] and len(direct.signed) == 1
    assert resumes(caplog) == [], "the download waited for a retry"
    [line] = switch_lines(caplog)
    assert "cannot be reached" in line and "ConnectionRefusedError" in line
    assert "X-Amz-Signature" not in caplog.text


def _full_listener():
    """A loopback port whose listen queue is full, so that the system drops
    new connections as a network that drops the packets does; None where it
    refuses them instead."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(0)
    held = []
    for _ in range(8):
        probe = socket.socket()
        probe.settimeout(0.3)
        try:
            probe.connect(server.getsockname())
        except TimeoutError:
            probe.close()
            return server, held
        except OSError:
            probe.close()
            break
        held.append(probe)
    for s in [server, *held]:
        s.close()
    return None


def test_storage_that_takes_no_connection_costs_the_connect_timeout_once(
    stub, storage, direct, tmp_path, monkeypatch, caplog
):
    """Storage whose packets are dropped: the env gives up after the connect
    timeout (10 s, 0.5 s here), not the minute a download waits for data,
    and streams from then on."""
    caplog.set_level(logging.INFO, logger="world_engine_client")
    full = _full_listener()
    if full is None:
        pytest.skip("this system refuses connections past a full listen queue")
    server, held = full
    monkeypatch.setattr(env_module, "_STORAGE_CONNECT_SECONDS", 0.5, raising=False)
    direct.base = f"http://127.0.0.1:{server.getsockname()[1]}"
    env = connect(stub)
    try:
        started = time.monotonic()
        assets = download(env, tmp_path)
        took = time.monotonic() - started
        for u in URLS:
            assert assets.read(u) == ASSETS[u]
    finally:
        env.close()
        for s in [server, *held]:
            s.close()
    assert asks(stub, URLS[0]) == [(True, None), (False, None)]
    assert all(asks(stub, u) == [(False, None)] for u in URLS[1:])
    assert 0.5 <= took < 10, took
    [line] = switch_lines(caplog)
    assert "cannot be reached" in line and "timed out" in line
    assert resumes(caplog) == []


# ----- storage's address ---------------------------------------------------- #
@pytest.mark.parametrize("host", ["::1", "2001:db8::abcd", "::ffff:127.0.0.1"])
def test_an_ipv6_address_without_a_port(stub, storage, direct, tmp_path, monkeypatch, caplog, host):
    """A URL with an IPv6 address and no port: the client connects to that
    address on port 80 (a resolver here sends it to storage) and names it in
    the Host header as the URL does."""
    caplog.set_level(logging.INFO, logger="world_engine_client")
    asked = resolve(monkeypatch, {host: [storage.server.server_address]})
    url = "/__asset/0.ply"
    direct.where[url] = f"http://[{host}]"
    env = connect(stub)
    try:
        assets = download(env, tmp_path, [url])
        assert assets.read(url) == ASSETS[url]
    finally:
        env.close()
    assert asked == [(host, 80)]
    [got] = storage.requests
    assert got.status == 200 and got.headers["Host"] == f"[{host}]"
    assert asks(stub, url) == [(True, None)]
    assert switch_lines(caplog) == [] and resumes(caplog) == []


# ----- what is not followed ------------------------------------------------- #
@pytest.fixture
def tls(tmp_path_factory):
    """A certificate authority, trusted for the test through SSL_CERT_FILE,
    and a server context for 127.0.0.1 signed by it."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.UTC)
    ca_key, key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-only CA")])
    usage = dict.fromkeys(
        (
            "content_commitment",
            "key_encipherment",
            "data_encipherment",
            "key_agreement",
            "encipher_only",
            "decipher_only",
        ),
        False,
    )

    def build(subject, public_key):
        return (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_name)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(hours=1))
            .not_valid_after(now + datetime.timedelta(days=1))
        )

    ca = (
        build(ca_name, ca_key.public_key())
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, **usage),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )
    leaf = (
        build(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]), key.public_key())
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(digital_signature=True, key_cert_sign=False, crl_sign=False, **usage),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    root = tmp_path_factory.mktemp("tls")
    (root / "ca.pem").write_bytes(ca.public_bytes(pem))
    (root / "server.pem").write_bytes(
        leaf.public_bytes(pem)
        + key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(root / "server.pem")
    return SimpleNamespace(ca=str(root / "ca.pem"), context=context)


def test_an_https_platform_is_not_followed_to_http(tls, tmp_path, caplog, monkeypatch):
    """The platform on https sends the splat to storage on https, which is
    followed, and the GLB to storage on http, which is not: that file and
    every one after it stream from the platform."""
    caplog.set_level(logging.INFO, logger="world_engine_client")
    monkeypatch.setenv("SSL_CERT_FILE", tls.ca)
    stub = start_stub(tls.context)
    secure, plain = Storage(tls.context), Storage()
    try:
        direct = Direct(stub, Platform(stub), secure)
        plain.objects.update(secure.objects)
        direct.where["/__asset/1.glb"] = plain.url
        env = connect(stub)
        try:
            assets = download(env, tmp_path)
            for u in URLS:
                assert assets.read(u) == ASSETS[u]
        finally:
            env.close()
        assert stub.url.startswith("https://") and secure.url.startswith("https://")
        [got] = secure.requests
        assert got.path == direct.object("/__asset/0.ply") and set(got.headers) <= OWN
        assert plain.requests == []
        assert asks(stub, "/__asset/1.glb") == [(True, None), (False, None)]
        for u in URLS[2:]:
            assert asks(stub, u) == [(False, None)]
        [line] = switch_lines(caplog)
        assert "an http URL while it is on https" in line
        assert resumes(caplog) == []
    finally:
        stub.close()
        secure.close()
        plain.close()


@pytest.mark.parametrize(
    "base",
    [
        None,
        "",
        "ftp://127.0.0.1",
        "http://user:pw@127.0.0.1:1",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:1/a b",
    ],
    ids=["no Location", "relative", "ftp", "user info", "port", "space"],
)
def test_a_url_this_client_does_not_follow_streams_at_once(
    stub, storage, direct, tmp_path, caplog, base
):
    """A redirect without a Location, or to a URL that is not an absolute
    http or https URL with a plain host and a path a request line can carry:
    the file streams from the platform at once, without a retry, and so does
    every file after it. The log line holds no part of the URL."""
    caplog.set_level(logging.INFO, logger="world_engine_client")
    url = "/__asset/0.ply"
    direct.where[url] = base
    env = connect(stub)
    try:
        assets = download(env, tmp_path, [url, "/__asset/1.glb"])
        assert assets.read(url) == ASSETS[url]
    finally:
        env.close()
    assert asks(stub, url) == [(True, None), (False, None)]
    assert asks(stub, "/__asset/1.glb") == [(False, None)]
    assert storage.requests == [] and resumes(caplog) == []
    [line] = switch_lines(caplog)
    assert "the platform redirected to an unusable URL" in line
    assert "X-Amz-Signature" not in caplog.text


def test_a_redirect_from_storage_is_not_followed(stub, storage, direct, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="world_engine_client")
    elsewhere = Storage()
    try:
        elsewhere.objects.update(storage.objects)
        url = "/__asset/0.ply"
        target = f"{elsewhere.url}{direct.object(url)}?X-Amz-Signature=elsewhere"
        storage.answers = [("redirect", target), ("redirect", target)]
        env = connect(stub)
        try:
            assets = download(env, tmp_path, [url, "/__asset/1.glb"])
            assert assets.read(url) == ASSETS[url]
        finally:
            env.close()
        assert elsewhere.requests == []
        # Each redirect from storage is a failure; the second one streams.
        assert [r.status for r in storage.requests] == [302, 302]
        assert asks(stub, url) == [(True, None), (True, None), (False, None)]
        assert asks(stub, "/__asset/1.glb") == [(False, None)]
        assert resumes(caplog) == []
        [line] = switch_lines(caplog)
        assert "answered HTTP 302" in line and "second failure" in line
    finally:
        elsewhere.close()


def test_only_scene_assets_ask_for_a_redirect_or_follow_one(stub, storage, direct):
    """The key, the capture page and the robot meshes never ask, and a
    redirect for one of them is an error, not followed; a scene asset's
    redirect is followed."""
    direct.also = {"/render/__key", "/render/index.html", "/render/__robot/0.bin"}
    env = connect(stub)
    try:
        for path in ("/__key", "/index.html", "/__robot/0.bin"):
            with pytest.raises(RenderFetchError) as caught:
                env._open_render(path)
            assert caught.value.status == 302
            [asked] = stub.get_calls("/render" + path)
            assert "X-We-Render-Redirect" not in asked
        assert storage.requests == []
        with env._open_render("/__asset/0.ply") as response:
            assert response.read() == direct.platform.cipher("/__asset/0.ply")
        assert [r.path for r in storage.requests] == [direct.object("/__asset/0.ply")]
    finally:
        env.close()


def test_a_redirect_the_client_did_not_ask_for_is_an_error(stub, storage, direct):
    """Once the env streams, the platform's redirect is not followed either."""
    direct.base = f"http://127.0.0.1:{free_port()}"
    env = connect(stub)
    try:
        with env._open_render("/__asset/0.ply") as response:  # storage out of reach
            assert response.read() == direct.platform.cipher("/__asset/0.ply")
        direct.base = storage.url
        elsewhere = f"{storage.url}{direct.object('/__asset/1.glb')}?X-Amz-Signature=x"
        stub.gets["/render/__asset/1.glb"] = lambda h: send_redirect(h, elsewhere)
        with pytest.raises(RenderFetchError) as caught:
            env._open_render("/__asset/1.glb")
        assert caught.value.status == 302
    finally:
        env.close()
    assert storage.requests == []
    assert asks(stub, "/__asset/0.ply") == [(True, None), (False, None)]
    assert asks(stub, "/__asset/1.glb") == [(False, None)]


# ----- what comes from storage is checked ----------------------------------- #
def _bad(platform: Platform, how: str) -> bytes:
    url = "/__asset/0.ply"
    cipher = platform.cipher(url)
    if how == "flipped byte":
        return cipher[:200] + bytes([cipher[200] ^ 1]) + cipher[201:]
    if how == "another key":
        other = ref.scene_key("another secret", "u", SCENE, "1.0")
        return ref.encrypt(other, platform.digests()[url], ASSETS[url])
    if how == "other bytes":  # every tag verifies, the plaintext is another file
        data, digest = ASSETS["/__asset/2.wem"], platform.digests()[url]
        head = ref.header(len(data), ref.CHUNK_SIZE, ref.key_id(platform.key()))
        gcm = ref.AesGcm(ref.file_key(platform.key(), digest, ref.CHUNK_SIZE))
        aad = ref._aad(hashlib.sha256(head).digest(), 0, True)
        return head + gcm.encrypt(ref._nonce(0), data, aad)
    return ASSETS[url]  # the plaintext


@pytest.mark.parametrize("how", ["flipped byte", "another key", "other bytes", "plaintext"])
def test_bytes_from_storage_are_checked_as_streamed_ones(stub, storage, direct, tmp_path, how):
    storage.answers = [("bytes", _bad(direct.platform, how))]
    env = connect(stub)
    try:
        with pytest.raises(wec.RendererError) as caught:
            download(env, tmp_path)
    finally:
        env.close()
    assert caught.value.code == "asset_mismatch"
    assert [r.status for r in storage.requests] == [200]
    assert blobs(tmp_path) == {}
    assert [p for p in Path(tmp_path, "tmp").rglob("*") if p.is_file()] == []
