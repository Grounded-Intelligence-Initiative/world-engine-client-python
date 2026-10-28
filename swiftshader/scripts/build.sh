#!/usr/bin/env bash
# Build the binaries of world-engine-swiftshader from pinned sources, inside the
# manylinux_2_28 image of this machine's architecture, and put them in
# src/world_engine_swiftshader/lib/:
#
#   libvulkan.so.1           the Khronos Vulkan loader (Vulkan-Loader, $VULKAN_SDK)
#   libvk_swiftshader.so     SwiftShader's Vulkan driver ($SWIFTSHADER_COMMIT); Reactor
#                            backend Subzero on x86_64, LLVM 10 on aarch64 (Subzero has no
#                            AArch64 backend)
#   vk_swiftshader_icd.json  its ICD manifest, library_path relative ("./libvk_swiftshader.so")
#   build_info.json          sources, backend, compiler, sizes, SHA-256, newest GLIBC symbol
#
# libstdc++ and libgcc are linked statically (SwiftShader's version script exports only the
# vk* symbols, so they never meet the host process's copies); the result needs glibc 2.28.
# Needs docker. Run on the machine whose architecture you build for (an arm64 runner for
# aarch64: no emulation).
#
#   swiftshader/scripts/build.sh [WORK_DIR] [JOBS]
#
# WORK_DIR (default: a new temporary directory) receives the source trees and the build.
# SWIFTSHADER_SRC=<dir> uses a SwiftShader tree already checked out at $SWIFTSHADER_COMMIT.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/sources.env"
ARCH=$(uname -m)
case "$ARCH" in
  x86_64) BACKEND=Subzero ;;
  aarch64) BACKEND=LLVM ;;
  *) echo "world-engine-swiftshader is built for x86_64 and aarch64, not $ARCH" >&2; exit 2 ;;
esac
WORK=${1:-$(mktemp -d)}
JOBS=${2:-$(nproc)}
IMAGE=${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_${ARCH}:latest}
OUT="$HERE/src/world_engine_swiftshader/lib"
mkdir -p "$WORK/src" "$WORK/out" "$OUT"

fetch() {  # fetch REPO REF DIR: one commit or tag, without history, then without .git
  local repo=$1 ref=$2 dir=$3
  [ -d "$dir" ] && return 0
  git init -q "$dir.git"
  git -C "$dir.git" remote add origin "$repo"
  git -C "$dir.git" fetch -q --depth 1 origin "$ref"
  git -C "$dir.git" -c advice.detachedHead=false checkout -q FETCH_HEAD
  rm -rf "$dir.git/.git"
  mv "$dir.git" "$dir"
}
if [ -n "${SWIFTSHADER_SRC:-}" ]; then
  SWS="$SWIFTSHADER_SRC"
else
  SWS="$WORK/src/swiftshader"
  fetch "$SWIFTSHADER_REPO" "$SWIFTSHADER_COMMIT" "$SWS" ||
    { rm -rf "$SWS.git"; fetch "$SWIFTSHADER_MIRROR" "$SWIFTSHADER_COMMIT" "$SWS"; }
fi
fetch https://github.com/KhronosGroup/Vulkan-Headers.git "$VULKAN_SDK" "$WORK/src/Vulkan-Headers"
fetch https://github.com/KhronosGroup/Vulkan-Loader.git "$VULKAN_SDK" "$WORK/src/Vulkan-Loader"

docker run --rm -e BACKEND="$BACKEND" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" \
  -v "$SWS":/sws:ro -v "$WORK/src":/src:ro -v "$WORK/out":/out "$IMAGE" bash -c '
set -euo pipefail
trap "chown -R $HOST_UID /out" EXIT
export PATH=/opt/python/cp312-cp312/bin:$PATH
pip install -q ninja==1.11.1.4
NINJA=$(command -v ninja)
cmake -S /src/Vulkan-Headers -B /out/hb -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA -DCMAKE_INSTALL_PREFIX=/out/headers > /out/headers.log 2>&1
cmake --install /out/hb >> /out/headers.log 2>&1
cmake -S /src/Vulkan-Loader -B /out/lb -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA -DCMAKE_BUILD_TYPE=Release \
  -DVULKAN_HEADERS_INSTALL_DIR=/out/headers -DCMAKE_SHARED_LINKER_FLAGS="-static-libgcc" \
  -DBUILD_WSI_XCB_SUPPORT=OFF -DBUILD_WSI_XLIB_SUPPORT=OFF -DBUILD_WSI_WAYLAND_SUPPORT=OFF \
  -DBUILD_WSI_DIRECTFB_SUPPORT=OFF > /out/loader.log 2>&1
ninja -C /out/lb -j $JOBS >> /out/loader.log 2>&1
cmake -S /sws -B /out/sb -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSWIFTSHADER_BUILD_TESTS=OFF -DSWIFTSHADER_BUILD_BENCHMARKS=OFF \
  -DSWIFTSHADER_BUILD_PVR=OFF -DSWIFTSHADER_WARNINGS_AS_ERRORS=OFF -DREACTOR_BACKEND=$BACKEND \
  -DSWIFTSHADER_BUILD_WSI_XCB=OFF -DSWIFTSHADER_BUILD_WSI_WAYLAND=OFF \
  -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc" > /out/swiftshader.log 2>&1
ninja -C /out/sb -j $JOBS vk_swiftshader >> /out/swiftshader.log 2>&1
mkdir -p /out/lib
cp "$(find /out/lb -name "libvulkan.so.1*" -type f | head -1)" /out/lib/libvulkan.so.1
cp "$(find /out/sb -name libvk_swiftshader.so -type f | head -1)" /out/lib/libvk_swiftshader.so
cp "$(find /out/sb -name vk_swiftshader_icd.json | head -1)" /out/lib/vk_swiftshader_icd.json
strip --strip-unneeded /out/lib/libvulkan.so.1 /out/lib/libvk_swiftshader.so
python3 - <<EOF
import json, hashlib, subprocess, re
info = {"backend": "$BACKEND", "gcc": subprocess.run(["gcc", "-dumpfullversion"], capture_output=True, text=True).stdout.strip(),
        "glibc": subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()[0], "files": {}}
for name in ("libvulkan.so.1", "libvk_swiftshader.so", "vk_swiftshader_icd.json"):
    p = "/out/lib/" + name
    data = open(p, "rb").read()
    entry = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if name.endswith(".so") or name.endswith(".so.1"):
        dyn = subprocess.run(["objdump", "-T", p], capture_output=True, text=True).stdout
        vers = sorted(set(re.findall(r"GLIBC_([0-9.]+)", dyn)), key=lambda v: tuple(map(int, v.split("."))))
        entry["max_glibc"] = vers[-1] if vers else None
        entry["glibcxx"] = sorted(set(re.findall(r"GLIBCXX_[0-9.]+", dyn)))
        entry["needed"] = re.findall(r"NEEDED\s+(\S+)", subprocess.run(["objdump", "-p", p], capture_output=True, text=True).stdout)
    info["files"][name] = entry
json.dump(info, open("/out/lib/build_info.part.json", "w"), indent=1)
EOF
'
python3 - "$WORK/out/lib/build_info.part.json" "$OUT/build_info.json" "$ARCH" <<EOF
import json, sys
info = json.load(open(sys.argv[1]))
info.update({"arch": sys.argv[3], "swiftshader": "$SWIFTSHADER_COMMIT", "vulkan_sdk": "$VULKAN_SDK", "image": "$IMAGE"})
json.dump(info, open(sys.argv[2], "w"), indent=1)
EOF
for f in libvulkan.so.1 libvk_swiftshader.so vk_swiftshader_icd.json; do cp "$WORK/out/lib/$f" "$OUT/$f"; done
python3 - "$OUT/vk_swiftshader_icd.json" <<'EOF'
import json, sys
icd = json.load(open(sys.argv[1]))
assert icd["ICD"]["library_path"] == "./libvk_swiftshader.so", icd
EOF
cat "$OUT/build_info.json"
