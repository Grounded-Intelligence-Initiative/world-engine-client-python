# world-engine-swiftshader

A software Vulkan for Linux, packaged for
[world-engine-client](https://pypi.org/project/world-engine-client/): Google's
SwiftShader, which runs Vulkan on the CPU, and the Khronos Vulkan loader.

world-engine-client renders camera images on your machine with wgpu. On Linux,
wgpu needs a Vulkan loader and a Vulkan driver. A machine without a GPU has no
driver, and minimal systems and many containers have no loader either. There,
world-engine-client renders with this package: slowly (about a step per second
on a real scene), but without installing system packages and without root.

You do not install it yourself: `pip install 'world-engine-client[render]'`
pulls it in on Linux x86_64 and aarch64. It is used only when nothing better
renders. The order is a GPU through the system's Vulkan, a GPU through this
package's loader, Mesa's lavapipe (`apt install mesa-vulkan-drivers`, faster
than this package), and last this package. It does nothing when imported.

## Contents

| file | what |
|---|---|
| `world_engine_swiftshader/lib/libvulkan.so.1` | the Khronos Vulkan loader |
| `world_engine_swiftshader/lib/libvk_swiftshader.so` | SwiftShader's Vulkan driver |
| `world_engine_swiftshader/lib/vk_swiftshader_icd.json` | its ICD manifest (relative library path) |
| `world_engine_swiftshader/lib/build_info.json` | sources, backend, compiler, sizes, SHA-256 |
| `world_engine_swiftshader/licenses/` | the licences of everything above (`NOTICE.md`) |

```python
import world_engine_swiftshader as sws
sws.loader_path(), sws.icd_path(), sws.build_info()
```

Wheels: `py3-none-manylinux_2_28_x86_64` and `py3-none-manylinux_2_28_aarch64`
(glibc 2.28 or newer: Debian 10, Ubuntu 18.10, RHEL 8 and later). There is no
source distribution: building needs Docker and about ten minutes.

## How it is built

From pinned sources (`sources.env`): SwiftShader `cddb5c91` and Vulkan-Loader and
Vulkan-Headers `vulkan-sdk-1.4.363.0`, in the `quay.io/pypa/manylinux_2_28` image of
the target machine, with libstdc++ and libgcc linked statically (SwiftShader
exports only its `vk*` symbols). SwiftShader's code generator is Subzero on x86_64
(a 2 MB wheel) and LLVM 10 on aarch64 (Subzero has no AArch64 backend; about 8 MB).

```bash
swiftshader/scripts/build.sh /tmp/sws-build     # on an x86_64 or an arm64 Linux machine with Docker
uv build --wheel swiftshader -o dist            # the wheel, tagged for that machine
python swiftshader/scripts/check_wheel.py dist/world_engine_swiftshader-*.whl --render
```

Releases are built and published by `.github/workflows/publish-swiftshader.yml`
in the world-engine-client-python repository, on a tag `swiftshader-v<version>`.

## Licences

SwiftShader, marl, SPIRV-Tools, astc-encoder and the Vulkan loader are
Apache-2.0; Subzero (x86_64) is NCSA; LLVM 10 (aarch64) is Apache-2.0 WITH
LLVM-exception plus legacy NCSA parts; SPIRV-Headers and cJSON are MIT. The
statically linked GCC runtime is covered by the GCC Runtime Library Exception.
The texts are in `world_engine_swiftshader/licenses/`.
