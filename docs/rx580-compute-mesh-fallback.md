# RX 580 compatibility: no-task mesh shaders via compute + indirect draws

Status: **partial research-branch implementation; compile validation is pending for the 3D-dispatch expansion, and RX 580 runtime validation has not happened.** The current code translates eligible direct mesh draws with no task stage into compute dispatches plus generated vertex data and conventional indirect draws. Direct 1D, 2D and 3D workgroup grids are supported by the fallback; task-stage and indirect mesh dispatches are not. This is not a complete task/mesh-shader replacement.

This note records the RX 580 failure mode and the fallback design. A capability check or environment-variable change cannot make native mesh shaders available on hardware that does not expose `VK_EXT_mesh_shader`.

## Evidence from the supplied RX 580 log

The user's log reports:

```
AMD Radeon RX 580 2048SP, Vulkan 1.3, driver 0x800117;
mesh shaders no, host memory import yes, pipeline libraries no
```

The Vulkan renderer starts successfully, so this is not a Vulkan-loader failure. The missing capability is specifically mesh/task shader execution.

In `gpu/vk/vk_main.cpp`, native mesh support is enabled only when `VK_EXT_mesh_shader` and both `meshShader` and `taskShader` features are present. On a device without those features, `mesh_program_locked()` rejects task-stage mesh programs but allows the no-task path to attempt compute emulation. The draw path dispatches the translated mesh stage as compute, writes a conventional vertex buffer and one indirect draw command per workgroup, then uses a generated vertex shader with the original fragment shader. Unsupported draws are skipped rather than sent to native mesh commands.

The WebGPU backend is not an equivalent workaround: it has no mesh stage and drops those draws, which can produce missing high-detail geometry and LOD pop-in.

## Separate issue in the same log: cooperative-matrix compute programs

The log also contains ten failed compute programs reporting:

```
not translated: a network left out on this card
rejected compute ...
```

This message is emitted by the cooperative-matrix policy in `gpu/vk/vk_shader.cpp`. That policy skips most of the game's cooperative-matrix movie-upscaler networks on non-NVIDIA GPUs because the software matrix-multiply path was prohibitively slow; selected networks are whitelisted separately.

This is a separate compute-shader policy, not proof of a mesh-shader failure. It should not be disabled wholesale as a mesh workaround. The supplied log alone does not establish that these ten failures explain the broken menu text; the menu/font issue needs its own capture after mesh draws render.

## Fallback architecture and remaining work

The branch translates an eligible no-task mesh workgroup into compute writes, then consumes those writes through a conventional vertex-input pipeline. It does not simply rename the mesh execution model to a vertex execution model.

### 1. Task stage remains unsupported

The current code does not translate task stages. `mesh_program_locked()` rejects any mesh program that has a task stage when native mesh shaders are unavailable. A future implementation must translate task-stage IR to compute, preserve ISBE payloads and barriers, compact each task's emitted mesh-workgroup count/culling decisions, and feed the surviving payload records into mesh compute without reading GPU results back to the CPU.

### 2. No-task mesh stage is partially implemented

The branch translates eligible no-task mesh stages to SPIR-V compute and writes vertex and indirect-command records through device addresses. Supported output is limited to line-list and triangle-list topology, position X/Y/Z, and generic varyings. Per-primitive generic values are duplicated to each generated corner and marked flat on the fragment inputs.

The compute emitter is prepared to flatten three-dimensional workgroup IDs into a unique output slot, but the current NVN replay maps the verified direct `DrawMeshTasks(first, count)` form to X only and fixes Y/Z to one. The raw third register in the captured record has not been confirmed as a group-count argument, so the current game-facing fallback remains 1D. The output packet index is flattened as:

```
x + dispatch_x * (y + dispatch_y * z) (with Y/Z currently 0 because the replay dispatches one-dimensional grids)
```

The shader-visible workgroup IDs are not flattened or rewritten, so shaders that read X/Y/Z continue to see the original grid coordinates. Indirect mesh dispatches, task payloads and unsupported output built-ins remain unsupported. Hardware/runtime validation is still outstanding.

### 3. Conventional indirect draws

The compute stage expands each output primitive to a line-list or triangle-list vertex sequence, so it does not need to build an index buffer. It writes one `VkDrawIndirectCommand` per workgroup. A generated vertex shader reads the output vertex buffer and forwards generic varyings to the original fragment shader. When `multiDrawIndirect` is unavailable, the renderer issues one indirect draw command at a time.

### 4. Synchronization, allocation and correctness

- The current draw path ends the render pass and inserts full barriers around the output clear and compute dispatch before consuming generated vertices and indirect commands. The barrier scopes still need validation on the target AMD driver.
- Output buffers are owned by the active recording; replaced buffers are retired and released when that recording slot is reused after completion. The growth path must continue to be checked for allocation failures and large draws.
- The draw path checks 32-bit output indexing, dispatch-axis limits, indirect-draw count limits and allocation failures; unsupported or oversized draws are logged and skipped.
- Remaining semantic gaps include task culling/payload, indirect mesh dispatches, viewport/layer outputs, primitive IDs, point size, clip/cull outputs and first-workgroup semantics where a task stage supplies them.
- Validate with scenes that exercise direct multidimensional no-task grids, task culling, large foliage/grass batches, multi-primitive meshlets and zero-output workgroups.

## Source areas that need coordinated changes

- `gpu/vk/vk_shader.cpp`: `mesh_program_locked`, `translate_mesh_into`, stage metadata and shader-module generation.
- `gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp` and `spirv_emit_context.*`: task/mesh epilogues, output interfaces and ISBE handling.
- `gpu/vk/vk_shader.h`: distinguish compute-emulated mesh programs from native mesh programs.
- `gpu/vk/vk_draw.cpp`: pipeline creation, descriptors, buffer binding, dispatch sequencing, indirect draw replay, resource lifetime and synchronization.
- `gpu/vk/vk_main.cpp` / `vk_internal.h`: capability selection; native mesh support must remain optional for the fallback.
- `nx2/build.py`: include any new translator/runtime source in the Vulkan DLL build.

The no-task fallback and direct 3D-dispatch expansion are source changes, not proof of compatibility. The Windows renderer compile check must pass, followed by actual RX 580 testing. Task-stage payload/culling translation remains the next major missing feature.

## Test plan for the RX 580

A meaningful test build should log a line identifying a no-task compute fallback, such as:

```
vk: mesh ... using no-task compute fallback
```

The test report should include:
1. Whether the title/menu text is legible.
2. Whether the player, nearby geometry, doors, fences and trees render completely.
3. Whether geometry pops/disappears when moving or changing camera distance.
4. Whether loading a save completes and whether the first playable area renders.
5. The GPU/driver line, fallback counters, skipped-draw counts and shader translation failures from the new log.

Keep a copy of the original game folder and saves before testing an experimental build. Do not set `RX_VK_FALLBACK_NO_MESH=1`; that opts out of Vulkan and hands rendering to WebGPU instead of testing this fallback.

## References

- [Vulkan mesh/task shader execution model](https://docs.vulkan.org/spec/latest/chapters/VK_NV_mesh_shader/mesh.html): task workgroups emit a variable number of mesh workgroups; mesh workgroups emit vertices and primitives.
- [VK_EXT_mesh_shader proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_mesh_shader.html): mesh shaders are compute-like workgroups and their outputs include vertex and primitive arrays.
- [Microsoft mesh-shader specification](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/MeshShader.md): describes compute-generated geometry as a traditional vertex/index-buffer fallback.
- Project source: `gpu/vk/vk_main.cpp`, `gpu/vk/vk_shader.cpp`, `gpu/vk/vk_draw.cpp` and `gpu/vk/VULKAN.md`.

## Release gate

Do not describe this as generally RX 580-compatible until:
- the supported no-task path passes a real render test on a device without mesh-shader support;
- native mesh calls are absent from the fallback draw path;
- the Vulkan DLL and Windows Builder package compile;
- a no-mesh-shader device successfully starts and renders a scene; and
- the RX 580 tester confirms the menu, save load and main scene.

This research note is not itself a compatibility patch or binary.
