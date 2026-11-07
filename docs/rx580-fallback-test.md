# RX 580 first compatibility test

This branch now includes an **opt-in WebGPU handoff** for systems without native Vulkan task/mesh shader support.

Set `RX_VK_FALLBACK_NO_MESH=1` in `hookshot_env.txt`. If Vulkan reports that `VK_EXT_mesh_shader`, `meshShader`, or `taskShader` is unavailable, the Vulkan DLL declines to start. The host's existing `rx_backend.c` then selects the WebGPU renderer.

This is a deliberately limited first test:
- It is an actual code path change and can be built as a Windows `hookshot_vk.dll`.
- It does **not** translate task/mesh shader programs into compute + vertex shaders.
- The existing WebGPU renderer does not emulate mesh output, so mesh-generated geometry can still be missing.
- It provides an A/B test of the non-mesh renderer path on the RX 580 and a fresh log before undertaking the full compute emulation.

The Windows workflow `.github/workflows/rx580-fallback-test.yml` builds only the renderer DLL (no game assets or user game data) and uploads `RX580-WebGPU-fallback-test.zip` as an Actions artifact. Install and rollback instructions are included in the zip.

Do not describe this test as an RX 580 fix unless the tester confirms the scene and interface render acceptably.