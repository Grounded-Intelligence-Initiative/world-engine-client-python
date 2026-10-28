"""The browser path's own browser, kept as the fallback of renderer D (design
doc 16 §8.1, the M-A2 items): Playwright pinned in the render extra; only its
headless shell, downloaded into a directory of this package next to the
render cache and never into Playwright's own; the download in the background,
beside the scene's; nothing on the terminal. No browser runs here: Playwright
is PageSim (test_render.py), its installer a stand-in."""

import hashlib
import io
import logging
import os
import re
import sys
import threading
import time
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_render import RTX, SCENE, PageSim, start

import world_engine_client as wec
from world_engine_client import cache
from world_engine_client import render as render_module

ROOT = Path(__file__).resolve().parents[1]
REV = "4242"  # the headless shell build the stand-in playwright names
NOT_FOUND = "Chromium distribution 'chrome' is not found at /opt/google/chrome/chrome"


def make_shell(root: Path, rev: str = REV) -> Path:
    """What Playwright leaves in ``root`` once the headless shell is in."""
    build = root / f"chromium_headless_shell-{rev}"
    exe = build / "chrome-headless-shell-linux64" / "chrome-headless-shell"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    (build / "INSTALLATION_COMPLETE").write_bytes(b"")
    return exe


@pytest.fixture
def sim(monkeypatch):
    """LocalRenderer with Playwright and websockets replaced by PageSim, as
    test_render's sim."""
    page = PageSim()
    monkeypatch.setattr(render_module, "available", lambda: True)
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(serve=page.serve))
    monkeypatch.setitem(sys.modules, "playwright", ModuleType("playwright"))
    async_api = SimpleNamespace(async_playwright=page.playwright)
    monkeypatch.setitem(sys.modules, "playwright.async_api", async_api)
    monkeypatch.delenv("WE_RENDER_CHROME_ARGS", raising=False)
    monkeypatch.setattr(render_module, "_slow_warned", False, raising=False)
    return page


@pytest.fixture
def shell_dir(tmp_path, monkeypatch):
    """This package's browsers directory, empty; a playwright that names REV;
    no download from an earlier test."""
    root = tmp_path / "browsers"
    monkeypatch.setenv("DIGIARENA_BROWSERS_DIR", str(root))
    monkeypatch.delenv("WE_RENDER_BROWSER", raising=False)
    monkeypatch.setattr(render_module, "_shell_revision", lambda: REV)
    monkeypatch.setattr(render_module, "_download", {"thread": None, "error": None})
    return root


@pytest.fixture
def downloads(shell_dir, monkeypatch):
    """Playwright's installer (subprocess.run), recorded: it puts the shell
    where PLAYWRIGHT_BROWSERS_PATH says, as Playwright does."""
    calls = []

    def run(cmd, **kw):
        calls.append({"cmd": list(cmd), **kw})
        make_shell(Path(kw["env"]["PLAYWRIGHT_BROWSERS_PATH"]))
        return SimpleNamespace(returncode=0, stdout=b"Chrome Headless Shell downloaded\n")

    monkeypatch.setattr(render_module.subprocess, "run", run)
    return calls


def no_chrome(sim, monkeypatch, *, shell_error: str | None = None):
    """No Google Chrome on this machine: Playwright's "chrome" channel fails.
    The shell launches (PageSim) unless ``shell_error``. Returns the launches,
    (channel, executable_path)."""
    monkeypatch.setattr(render_module, "chrome_installed", lambda: False)
    launches = []
    launch = sim.launch

    async def without_chrome(channel=None, args=(), executable_path=None, **kw):
        launches.append((channel, executable_path))
        if channel == "chrome":
            raise RuntimeError(NOT_FOUND)
        assert executable_path, "Playwright's own browser directory is never used"
        if shell_error:
            raise RuntimeError(shell_error)
        browser = await launch(channel, args, **kw)
        browser.version = "148.0.7778.96"
        return browser

    monkeypatch.setattr(sim, "launch", without_chrome)
    return launches


# ------------------------------------------------------------------- the pin
def test_the_render_extra_pins_playwright():
    """Which headless shell downloads follows playwright's version: a range
    gives each new install another Chromium (design doc 16 §8.1)."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    render = project["optional-dependencies"]["render"]
    named = [s.replace(" ", "") for s in render if re.match(r"playwright\b", s.strip(), re.I)]
    assert len(named) == 1 and re.fullmatch(r"playwright==\d+\.\d+\.\d+", named[0]), named
    assert any(s.strip().startswith("websockets") for s in render), "the browser path's socket"


# ------------------------------------------------- only the shell, our directory
def test_only_the_headless_shell_downloads_into_this_packages_directory(
    downloads, tmp_path, monkeypatch
):
    monkeypatch.delenv("DIGIARENA_BROWSERS_DIR")
    monkeypatch.delenv("DIGIARENA_CACHE_DIR", raising=False)
    monkeypatch.setattr(cache, "user_cache_dir", lambda: tmp_path / "user")
    before = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    exe = render_module.install_chromium()
    root = render_module.browsers_dir()
    assert root == tmp_path / "user" / "digiarena" / "browsers"
    assert root.parent == cache.directory().parent, "next to the render cache"
    (call,) = downloads
    installer = [sys.executable, "-m", "playwright", "install", "--only-shell", "chromium"]
    assert call["cmd"] == installer
    assert call["env"]["PLAYWRIGHT_BROWSERS_PATH"] == str(root)
    assert os.environ.get("PLAYWRIGHT_BROWSERS_PATH") == before, "this process keeps its own"
    assert exe == render_module.shell_executable() and exe.is_relative_to(root)
    assert render_module.install_chromium() == exe and len(downloads) == 1, "it downloads once"


def test_a_download_that_stopped_part_way_is_not_a_shell(shell_dir):
    exe = make_shell(shell_dir)
    (exe.parent.parent / "INSTALLATION_COMPLETE").unlink()
    assert render_module.shell_executable() is None
    make_shell(shell_dir, rev="1111")  # another playwright's build
    assert render_module.shell_executable() is None


def test_the_shell_is_launched_from_this_packages_directory(sim, downloads, monkeypatch):
    launches = no_chrome(sim, monkeypatch)
    sim.gpu = RTX
    renderer = start(sim, timeout=10, auto_install=True)
    try:
        exe = render_module.shell_executable()
        assert exe is not None and launches == [("chrome", None), (None, str(exe))]
        assert renderer.info["browser"] == "headless-shell"
        assert renderer.info["browser_version"] == "148.0.7778.96"
        assert len(downloads) == 1
    finally:
        renderer.close()


def test_an_installed_chrome_comes_first_and_says_so(sim, downloads, monkeypatch):
    monkeypatch.setattr(render_module, "chrome_installed", lambda: True)
    sim.gpu = RTX
    renderer = start(sim, timeout=10, auto_install=True)
    try:
        assert renderer.info["browser"] == "chrome"
        assert downloads == [], "no download where Chrome is installed"
    finally:
        renderer.close()


# ------------------------------------------------------------ in the background
def test_the_shell_downloads_while_the_scene_does(sim, shell_dir, monkeypatch):
    """No Chrome and no shell yet: the shell's download starts before the
    scene's and runs beside it. One after the other, each would wait here for
    the other to begin, and the start fails."""
    no_chrome(sim, monkeypatch)
    sim.gpu = RTX
    began = {"shell": threading.Event(), "scene": threading.Event()}

    def run(cmd, **kw):
        began["shell"].set()
        assert began["scene"].wait(10), "the scene did not download while the shell did"
        make_shell(Path(kw["env"]["PLAYWRIGHT_BROWSERS_PATH"]))
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(render_module.subprocess, "run", run)
    data = b"\x00splat" * 4096
    digests = {"/__asset/0.ply": hashlib.sha256(data).hexdigest()}

    def fetch(path, headers=None):
        if path not in digests:
            raise FileNotFoundError(path)
        began["scene"].set()
        assert began["shell"].wait(10), "the shell's download had not begun"
        return io.BytesIO(data)

    renderer = render_module.LocalRenderer(fetch, SCENE, digests, timeout=10, auto_install=True)
    try:
        assert renderer.info["browser"] == "headless-shell"
    finally:
        renderer.close()


@pytest.mark.parametrize("case", ["chrome installed", "shell there", "download off", "chrome only"])
def test_no_download_starts_when_none_is_needed(sim, downloads, monkeypatch, case):
    monkeypatch.setattr(render_module, "chrome_installed", lambda: case == "chrome installed")
    if case == "shell there":
        make_shell(render_module.browsers_dir())
    if case == "chrome only":
        monkeypatch.setenv("WE_RENDER_BROWSER", "chrome")
    renderer = start(sim, timeout=10, auto_install=case != "download off")
    renderer.close()
    assert downloads == [] and render_module._download["thread"] is None


def test_downloading_the_shell_does_not_count_against_the_start(sim, shell_dir, monkeypatch):
    monkeypatch.setattr(render_module, "_START_GRACE", 0.3, raising=False)
    no_chrome(sim, monkeypatch)
    sim.gpu = RTX

    def run(cmd, **kw):
        time.sleep(2)  # longer than timeout + _START_GRACE
        make_shell(Path(kw["env"]["PLAYWRIGHT_BROWSERS_PATH"]))
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(render_module.subprocess, "run", run)
    renderer = start(sim, timeout=1, auto_install=True)
    try:
        assert renderer.info["browser"] == "headless-shell" and not renderer.software
    finally:
        renderer.close()


# --------------------------------------------------------- nothing on the terminal
FAKE_INSTALLER = """
import os, sys
from pathlib import Path
print("Chrome Headless Shell 148.0.7778.96 downloaded |######| 100% of 113.2 MiB", flush=True)
print("a line on stderr", file=sys.stderr, flush=True)
assert sys.argv[1:] == ["install", "--only-shell", "chromium"], sys.argv
build = Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"]) / "chromium_headless_shell-4242"
exe = build / "chrome-headless-shell-linux64" / "chrome-headless-shell"
exe.parent.mkdir(parents=True)
exe.write_text("")
(build / "INSTALLATION_COMPLETE").write_text("")
"""


def test_the_download_prints_nothing_on_the_terminal(
    shell_dir, tmp_path, monkeypatch, capfd, caplog
):
    """A real installer process (a stand-in playwright on PYTHONPATH that
    writes to both of its streams): nothing reaches this process's stdout or
    stderr; the download and the installer's output are in the log."""
    fake = tmp_path / "fake-playwright" / "playwright"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    (fake / "__main__.py").write_text(FAKE_INSTALLER)
    monkeypatch.setenv("PYTHONPATH", str(fake.parent))
    caplog.set_level(logging.DEBUG, logger="world_engine_client.render")
    capfd.readouterr()
    exe = render_module.install_chromium()
    out, err = capfd.readouterr()
    assert (out, err) == ("", ""), "nothing on the terminal"
    assert exe == render_module.shell_executable()
    logged = [r.getMessage() for r in caplog.records if r.name == "world_engine_client.render"]
    assert any("downloading the headless browser" in m for m in logged), logged
    assert any("100% of 113.2 MiB" in m for m in logged), "the installer's output is logged"
    assert any("a line on stderr" in m for m in logged), logged


# --------------------------------------------------------------- when it fails
def test_a_failed_download_is_tried_once_and_says_how_to_fix_it(sim, shell_dir, monkeypatch):
    """Once per start, for all of Chrome's flag sets, and its fixes reach
    the error the user sees."""
    no_chrome(sim, monkeypatch)
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=1, stdout=b"Error: getaddrinfo ENOTFOUND cdn.example\n")

    monkeypatch.setattr(render_module.subprocess, "run", run)
    with pytest.raises(wec.RendererError) as caught:
        start(sim, timeout=10, auto_install=True)
    text = str(caught.value)
    assert caught.value.code == "renderer_unavailable"
    assert "ENOTFOUND" in text and "python -m world_engine_client.render install" in text
    assert "HTTPS_PROXY" in text and "images=False" in text
    assert len(calls) == 1, "one download for all flag sets"


def test_with_the_download_off_the_error_says_how_to_install(sim, downloads, monkeypatch):
    no_chrome(sim, monkeypatch)
    with pytest.raises(wec.RendererError) as caught:
        start(sim, timeout=10, auto_install=False)
    text = str(caught.value)
    assert "no headless browser in" in text and "automatic download is off" in text
    assert "python -m world_engine_client.render install" in text
    assert downloads == []


def test_missing_system_libraries_point_to_install_deps(sim, shell_dir, monkeypatch):
    make_shell(shell_dir)
    broken = "Host system is missing dependencies to run browsers. libnss3.so"
    no_chrome(sim, monkeypatch, shell_error=broken)
    with pytest.raises(wec.RendererError) as caught:
        start(sim, timeout=10, auto_install=True)
    assert caught.value.fixes == [render_module._SYSTEM_DEPS_FIX]
    assert "install-deps" in str(caught.value) and "--with-deps" not in str(caught.value)


def test_a_browser_that_fails_with_one_flag_set_gets_the_next(sim, shell_dir, monkeypatch):
    """Only a missing browser ends the start at once: a shell that is there
    and does not start with one flag set is launched with the next."""
    make_shell(shell_dir)
    no_chrome(sim, monkeypatch)
    sim.gpu = RTX
    launch = sim.launch
    tried = []

    async def first_set_crashes(channel=None, args=(), executable_path=None, **kw):
        tried.append(list(args))
        if len(tried) == 2:  # the shell with the first flag set (Chrome was the first try)
            raise RuntimeError("Browser closed: the GPU process crashed")
        return await launch(channel, args, executable_path=executable_path, **kw)

    monkeypatch.setattr(sim, "launch", first_set_crashes)
    renderer = start(sim, timeout=10, auto_install=True)
    try:
        assert renderer.chrome_args == render_module.candidate_args()[1]
        assert renderer.info["browser"] == "headless-shell"
    finally:
        renderer.close()


def test_auto_install_follows_the_option_then_the_environment(monkeypatch):
    monkeypatch.delenv("WE_RENDER_AUTO_INSTALL", raising=False)
    assert render_module._auto_install_enabled(None) is True
    monkeypatch.setenv("WE_RENDER_AUTO_INSTALL", "0")
    assert render_module._auto_install_enabled(None) is False
    assert render_module._auto_install_enabled(True) is True


# ------------------------------------------------------------ WE_RENDER_BROWSER
def test_we_render_browser_shell_skips_chrome(sim, downloads, monkeypatch):
    launches = no_chrome(sim, monkeypatch)
    monkeypatch.setattr(render_module, "chrome_installed", lambda: True)
    monkeypatch.setenv("WE_RENDER_BROWSER", "shell")
    sim.gpu = RTX
    renderer = start(sim, timeout=10, auto_install=True)
    try:
        assert [c for c, _ in launches] == [None] and renderer.info["browser"] == "headless-shell"
    finally:
        renderer.close()


def test_we_render_browser_chrome_never_downloads(sim, downloads, monkeypatch):
    no_chrome(sim, monkeypatch)
    monkeypatch.setenv("WE_RENDER_BROWSER", "chrome")
    with pytest.raises(wec.RendererError, match="install Google Chrome"):
        start(sim, timeout=10, auto_install=True)
    assert downloads == []


def test_we_render_browser_takes_three_values(monkeypatch):
    for value in ("auto", "chrome", "shell", " Shell "):
        monkeypatch.setenv("WE_RENDER_BROWSER", value)
        assert render_module.browser_choice() == value.strip().lower()
    monkeypatch.setenv("WE_RENDER_BROWSER", "firefox")
    with pytest.raises(ValueError, match="WE_RENDER_BROWSER"):
        render_module.browser_choice()


# ---------------------------------------------------------------- where and how
def test_chrome_is_looked_for_where_playwright_looks():
    """Playwright's chrome channel (its registry, 1.60.0): one path on Linux and
    macOS; on Windows under LOCALAPPDATA, PROGRAMFILES, PROGRAMFILES(X86) and
    HOMEDRIVE's Program Files."""
    paths = render_module._chrome_paths
    assert paths("linux", {}) == ["/opt/google/chrome/chrome"]
    assert paths("darwin", {}) == ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    env = {"LOCALAPPDATA": "C:\\Users\\u\\AppData\\Local", "PROGRAMFILES": "C:\\Program Files"}
    env["HOMEDRIVE"] = "D:"
    assert paths("win32", env) == [
        "C:\\Users\\u\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "D:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        "D:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    ]


def test_the_command_line_installs_and_says_where(downloads, capsys):
    assert render_module.main(["where"]) == 0
    where = capsys.readouterr().out
    assert str(render_module.browsers_dir()) in where and '"shell": null' in where
    assert render_module.main(["install"]) == 0
    assert capsys.readouterr().out.strip() == str(render_module.shell_executable())
    assert len(downloads) == 1


def test_the_real_playwright_names_its_shell_build():
    """The pinned playwright says which shell build it downloads: the
    directory name shell_executable looks for."""
    pytest.importorskip("playwright")
    rev = render_module._shell_revision()
    assert rev is not None and rev.isdigit(), rev


def test_info_names_the_browser(sim):
    sim.gpu = RTX
    renderer = start(sim, timeout=10)
    try:
        info = renderer.info
        assert info["renderer"] == "chrome" and info["adapter"] == RTX
        assert info["browser"] in ("chrome", "headless-shell")
        assert "browser_version" in info
    finally:
        renderer.close()
