"""A session's scene assets in memory, for the native renderer (renderer D).

:class:`world_engine_client.cache.SceneAssets` keeps the assets on disk as they
arrived: DAC1 ciphertext from the hosting platform, plain files from your own
local ``world-engine serve``. :func:`read` decrypts one of them into memory and
checks it against its digest; the renderer parses the bytes from there
(``world_engine_client.wgsl.splat_data`` views a PLY with a numpy structured
dtype, ``wgsl.scene`` reads GLB meshes and robot links) and uploads them to the
GPU. No plaintext is written to a file at any point; the buffer is dropped once
it is on the GPU.
"""

from __future__ import annotations

from .cache import SceneAssets, _Mismatch
from .dac1 import Dac1Error
from .errors import RendererError


def read(assets: SceneAssets, url: str) -> memoryview:
    """The verified plaintext of one of the session's assets, in one buffer of
    exactly its length (no second copy while it is assembled)."""
    try:
        length = assets.length(url)
    except KeyError:
        raise RendererError(
            f"the scene names an asset the server did not list: {url}", code="asset_mismatch"
        ) from None
    buf = bytearray(length)
    pos = 0
    try:
        for piece in assets.open(url):
            end = pos + len(piece)
            if end > length:
                raise _Mismatch("longer than its recorded length")
            buf[pos:end] = piece
            pos = end
    except (Dac1Error, _Mismatch) as exc:
        raise RendererError(
            f"the cached asset {url} does not match its digest: {exc}", code="asset_mismatch"
        ) from None
    except OSError as exc:  # the cache file went away or the disk failed
        raise RendererError(
            f"reading {url} from the render cache failed: {exc}",
            code="renderer_unavailable",
            fixes=["connect again (the asset is downloaded again)"],
        ) from None
    if pos != length:
        raise RendererError(
            f"the cached asset {url} ended after {pos} of {length} bytes", code="asset_mismatch"
        )
    return memoryview(buf)
