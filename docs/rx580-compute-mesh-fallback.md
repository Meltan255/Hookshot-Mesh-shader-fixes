# RX 580 compatibility: no-task mesh shaders via compute + indirect draws

Status: **experimental task+mesh compute path is pushed; Windows Vulkan renderer compilation and Builder ZIP packaging passed at test 38. RX 580 runtime/visual validation is still pending.** No-task mesh draws and direct task+mesh draws are translated to compute, generated vertices and conventional indirect draws. The current NVN replay path maps the verified direct `DrawMeshTasks(first, count)` form to X only and fixes Y/Z to one. Indirect mesh/task draws remain unsupported. This is not a complete task/mesh-shader replacement.

This note records the RX 580 failure mode and the fallback design. A capability check or environment-variable change cannot make native mesh shaders available on hardware that does not expose `VK_EXT_mesh_shader`.

## Evidence from the supplied RX 580 log

The user's log reports:

```
AMD Radeon RX 580 2048SP, Vulkan 1.3, driver 0x800117;
mesh shaders no, host memory import yes, pipeline libraries no
```

The Vulkan renderer starts successfully, so this is not a Vulkan-loader failure. The missing capability is specifically mesh/task shader execution.

In `gpu/vk/vk_main.cpp`, native mesh support is enabled only when `VK_EXT_mesh_shader` and both `meshShader` and `taskShader` features are present. On a device without those features, the research branch now attempts compute emulation for both task and mesh stages; this new path is experimental and still needs a successful Windows renderer build and RX 580 run. The draw path dispatches the translated mesh stage as compute, writes a conventional vertex buffer and one indirect draw command per workgroup, then uses a generated vertex shader with the original fragment shader. Unsupported draws are skipped rather than sent to native mesh commands.

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

### 1. Task stage is now experimentally translated to compute

For a direct one-dimensional task draw, the research branch translates the NVN task stage to a compute shader. The task stage preserves its guest-visible `WorkgroupId.x` by adding `firstTask`, then writes one GPU record per task workgroup: the clamped emitted-child count followed by the task payload words. The payload record is stored through a device address in a buffer owned by the active recording slot.

The renderer dispatches task compute first, inserts a compute-write to compute-read barrier, and dispatches mesh compute over a fixed maximum fan-out. The maximum children per task defaults to that task shader's X workgroup size; `RX_VK_TASK_CAP` can lower or raise the bound. Each mesh workgroup maps its linear dispatch slot back to a parent task and child ordinal, copies the parent's payload into workgroup memory, and exposes the child ordinal as the mesh shader's guest-visible `WorkgroupId.x`. Child slots beyond the parent's emitted count write an empty indirect draw.

This preserves GPU-side culling and avoids CPU readback, but it is not identical to native variable-size task dispatch: the mesh shader body still runs for over-dispatched empty child slots, even though their output primitive count is forced to zero. To avoid executing unintended external writes from those extra slots, for task+mesh draws, the translator rejects mesh programs with external guest-memory, storage-buffer, image-write or atomic side effects, because those child slots can be over-dispatched. No-task mesh draws do not use this fixed-fan-out restriction. Workgroup-local scratch and the internal mesh payload/output writes remain allowed. This is intentionally conservative and can reject shaders that might otherwise be safe. Only direct one-dimensional task draws are accepted; indirect task draws and multidimensional task grids are rejected.

The task and mesh payload word counts are checked for equality, task fan-out and Vulkan dispatch limits are checked, and the payload records live until the recording slot is safely reused. The firstTask offset is applied to the task shader's guest-visible WorkgroupId.x; the mesh shader sees its child ordinal within the parent task. Subgroup capability checks for emulated task and mesh shaders use compute-stage support, because the generated SPIR-V executes as GLCompute. The Windows renderer compile passed, but the implementation remains experimental and has not been tested on RX 580 hardware.

### 2. No-task mesh stage is partially implemented

The branch translates eligible no-task mesh stages to SPIR-V compute and writes vertex and indirect-command records through device addresses. Supported output is limited to line-list and triangle-list topology, position X/Y/Z, and generic varyings. Per-primitive generic values are duplicated to each generated corner and marked flat on the fragment inputs.

The compute emitter is prepared to flatten three-dimensional workgroup IDs into a unique output slot, but the current NVN replay maps the verified direct `DrawMeshTasks(first, count)` form to X only and fixes Y/Z to one. The raw third register in the captured record has not been confirmed as a group-count argument, so the current game-facing fallback remains 1D. For that direct no-task path, the compute shader adds the recorded `firstTask` to the guest-visible X workgroup ID, while the output epilogue keeps the local workgroup ID so its generated-vertex writes stay within the per-draw allocation. The output packet index is flattened as:

```
x + dispatch_x * (y + dispatch_y * z) (with Y/Z currently 0 because the replay dispatches one-dimensional grids)
```

The output packet index is not the shader-visible workgroup ID: X remains `firstTask + local WorkgroupId.x`, while the local Y/Z components are preserved for a dispatch grid that supplies them. In the task path, the task shader receives `firstTask + task WorkgroupId.x`; child mesh shaders receive their child ordinal within the parent task. Indirect mesh dispatches, indirect task dispatches and unsupported output built-ins remain unsupported. Hardware/runtime validation is still outstanding.

### 3. Conventional indirect draws

The compute stage expands each output primitive to a line-list or triangle-list vertex sequence, so it does not need to build an index buffer. It writes one `VkDrawIndirectCommand` per workgroup. A generated vertex shader reads the output vertex buffer and forwards generic varyings to the original fragment shader. When `multiDrawIndirect` is unavailable, the renderer issues one indirect draw command at a time.

### 4. Synchronization, allocation and correctness

- The current draw path ends the render pass and inserts full barriers around the output clear and compute dispatch before consuming generated vertices and indirect commands. The barrier scopes still need validation on the target AMD driver.
- Output buffers are owned by the active recording; replaced buffers are retired and released when that recording slot is reused after completion. The growth path must continue to be checked for allocation failures and large draws.
- The draw path checks 32-bit output indexing, dispatch-axis limits, indirect-draw count limits and allocation failures; unsupported or oversized draws are logged and skipped.
- Remaining semantic gaps include indirect mesh/task dispatches, performance cost from over-dispatched empty child slots, rejected task/mesh programs with external side effects, viewport/layer outputs, primitive IDs, point size and clip/cull outputs.
- Validate with scenes that exercise direct no-task draws with nonzero firstTask, task culling, large foliage/grass batches, multi-primitive meshlets and zero-output workgroups.

## Source areas that need coordinated changes

- `gpu/vk/vk_shader.cpp`: `mesh_program_locked`, `translate_mesh_into`, stage metadata and shader-module generation.
- `gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp` and `spirv_emit_context.*`: task/mesh epilogues, output interfaces and ISBE handling.
- `gpu/vk/vk_shader.h`: distinguish compute-emulated mesh programs from native mesh programs.
- `gpu/vk/vk_draw.cpp`: pipeline creation, descriptors, buffer binding, dispatch sequencing, indirect draw replay, resource lifetime and synchronization.
- `gpu/vk/vk_main.cpp` / `vk_internal.h`: capability selection; native mesh support must remain optional for the fallback.
- `nx2/build.py`: include any new translator/runtime source in the Vulkan DLL build.

The task+mesh fallback, no-task fallback and dispatch/indexing changes are source changes, not proof of compatibility. The Windows Vulkan renderer compile and Builder ZIP package passed at test 38; actual RX 580 testing is still required. The next major work is supporting indirect task/mesh dispatches, unsupported mesh outputs and a more exact variable-child execution model.

## Test plan for the RX 580

A meaningful test build should log a line identifying compute mesh emulation, such as:

```
vk: mesh ... using compute mesh fallback
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
- the direct no-task and task+mesh compute paths pass a real render test on a device without mesh-shader support;
- native mesh calls are absent from the fallback draw path;
- the Vulkan DLL and Windows Builder package compile;
- a no-mesh-shader device successfully starts and renders a scene; and
- the RX 580 tester confirms the menu, save load and main scene.

This research note is not itself a compatibility patch or binary.
