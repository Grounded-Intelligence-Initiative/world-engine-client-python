# Licences of the files in world-engine-swiftshader

Built from SwiftShader cddb5c91b23ff70a1f607ea31610aa0d7f4b5ad5 (2026-10-14) and
Vulkan-Loader and Vulkan-Headers vulkan-sdk-1.4.363.0. Every licence below allows
redistribution; the obligation is to ship these notices.

| file | component | licence | text |
|---|---|---|---|
| lib/libvk_swiftshader.so | SwiftShader (Vulkan driver, Reactor) | Apache-2.0 | SwiftShader-LICENSE.txt |
| | marl | Apache-2.0 | marl-LICENSE.txt |
| | SPIRV-Tools | Apache-2.0 | SPIRV-Tools-LICENSE.txt |
| | SPIRV-Headers (compiled in) | MIT (Khronos) | SPIRV-Headers-LICENSE.txt |
| | astc-encoder | Apache-2.0 | astc-encoder-LICENSE.txt |
| | Subzero and llvm-subzero (the x86_64 wheels' Reactor backend) | NCSA | Subzero-LICENSE.TXT, llvm-subzero-LICENSE.TXT |
| | LLVM 10 (the aarch64 wheels' Reactor backend) | Apache-2.0 WITH LLVM-exception, and the legacy NCSA licence (some files still carry the University of Illinois header; include/llvm/Support/LICENSE.TXT adds an NCSA notice) | LLVM-10.0.0-LICENSE.TXT, LLVM-Support-SystemInterface-LICENSE.TXT |
| | libstdc++ and libgcc (linked statically) | GPL-3.0-or-later WITH GCC-exception-3.1 | none needed: the GCC Runtime Library Exception puts no condition on the combined work |
| lib/libvulkan.so.1 | Khronos Vulkan-Loader | Apache-2.0 | Vulkan-Loader-LICENSE.txt |
| | cJSON (in the loader) | MIT | cJSON-LICENSE.txt |
| | Vulkan-Headers (compiled in) | Apache-2.0 OR MIT | Vulkan-Headers-LICENSE.md |

Each wheel carries every text above; an x86_64 wheel contains no LLVM 10 code and an
aarch64 wheel no Subzero code (lib/build_info.json names the backend).
