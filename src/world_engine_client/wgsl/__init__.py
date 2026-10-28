"""Renderer D's Python host (design doc 16 §9): the WGSL splat and mesh passes on wgpu-py.

``render_wgsl`` (the renderers: ``SplatRenderer`` takes the parts,
``WgslRenderer`` a session's scene_init and its assets), ``splat_data`` (PLY
and GPU layouts), ``mesh_data`` (WEM1 and WEL1 meshes, three's material rules,
textures), ``lighting`` (the capture page's lighting set-ups, HDRI readers,
WELI1 assets), ``mesh_pass`` (mesh, shadow and prefilter GPU resources),
``budget`` (memory plan), ``adapters`` (the adapter search: GPU, then
software; ``_probe`` runs one step of it in a subprocess). They import nothing
from the engine or from world-engine-client, only numpy, wgpu and Pillow.

The engine repository is the one source of these files and of the shaders
(``frontend/src/render/wgsl/``). world-engine-client ships a copy of both in its
package ``world_engine_client.wgsl`` (its ``scripts/sync_wgsl.py`` copies them
and checks the copy), so local ``we.make(renderer="wgsl")`` and the Python client
run the same host. Importing this package does not import wgpu; constructing a
renderer or opening an adapter does.
"""
