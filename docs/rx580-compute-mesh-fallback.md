# RX 580 compatibility research: task/mesh shaders via compute + indexed draws

Status: **design/research only — not an implementation or a test build.**

This note records the RX 580 failure mode and a concrete fallback design. It deliberately does not claim that a capability check or an environment-variable change can make mesh shaders work on hardware that does not expose `VK_EXT_mesh_shader`.

## Evidence from the supplied RX 580 log

The user's log reports:

```
AMD Radeon RX 580 2048SP, Vulkan 1.3, driver 0x800117;
mesh shaders no, host memory import yes, pipeline libraries no
```

The Vulkan renderer starts successfully, so this is not a Vulkan-loader failure. The missing capability is specifically mesh/task shader execution.

In `gpu/vk/vk_shader.cpp`, `mesh_program_locked()` returns `nullptr` immediately when `dev.mesh_shader` is false. In `gpu/vk/vk_main.cpp`, that flag is set only when the optional `VK_EXT_mesh_shader` extension and both `meshShader` and `taskShader` features are present. The normal draw pipeline in `gpu/vk/vk_draw.cpp` treats mesh programs specially: it omits vertex-input and input-assembly state and dispatches with `vkCmdDrawMeshTasksEXT` / its indirect form. That path cannot be used by this RX 580.

The WebGPU backend is not an equivalent workaround: `gpu/vk/VULKAN.md` explicitly says the WebGPU renderer has no mesh stage and drops those draws, which produces missing high-detail geometry / LOD pop-in.

## Separate issue in the same log: cooperative-matrix compute programs

The log also contains ten failed compute programs reporting:

```
not translated: a network left out on this card
rejected compute ...
```

This message is emitted by the cooperative-matrix policy in `gpu/vk/vk_shader.cpp`. The surrounding source comment says this policy skips most of the game's cooperative-matrix movie-upscaler networks on non-NVIDIA GPUs because their software matrix multiply path was prohibitively slow; two explicitly whitelisted networks are retained, including one associated with the dynamic-resolution scene.

This is a separate compute-shader policy, not proof of a mesh-shader failure. It should not be disabled wholesale as a mesh workaround. The supplied log alone does not establish that these ten failures explain the broken menu text; the menu/font issue needs its own capture after a real mesh fallback exists.

## Proposed fallback architecture

Use the GPU's ordinary compute and graphics stages to emulate the NVN2 task/mesh work. Do not attempt to rename a mesh execution model to a vertex execution model: mesh shaders are workgroup programs with shared memory, barriers, task payloads, variable output counts, per-primitive data and generated indices.

### 1. Task stage becomes a compute dispatch

- Translate NVN task-stage IR to a SPIR-V compute module, preserving workgroup size, shared memory, barriers, global-memory accesses and the NVN internal-stage-buffer (ISBE) payload semantics.
- Replace the `OpEmitMeshTasksEXT` marker currently emitted by `TaskEpilogue` with writes to a per-task record: emitted child-workgroup count plus the payload for each child.
- Keep task output GPU-resident. Do not read counts back to the CPU.
- Produce a bounded/validated work list or indirect-dispatch dimensions. A task may emit zero workgroups, and counts must be clamped to the game's declared limits and device/storage-buffer limits.

### 2. Mesh stage becomes a compute dispatch

- Translate the NVN mesh-stage IR to SPIR-V compute, preserving workgroup IDs, local invocation IDs, subgroup behavior, shared-memory semantics and barriers.
- Replace the `OpSetMeshOutputsEXT` / mesh output interface with storage-buffer writes.
- For each mesh workgroup, write its vertex attributes, primitive indices, per-primitive attributes and actual vertex/primitive counts into a uniquely assigned output range.
- Preserve the distinction between per-vertex and per-primitive outputs. Preserve points/lines/triangles as list topologies; do not assume every output is a triangle.
- Carry the task's payload and original workgroup identity into the mesh compute dispatch. A rectangular over-dispatch with inactive-workgroup guards is one possible first implementation; an indirect compacted work list is preferable if dispatch limits or wasted work make that unsuitable.

### 3. Build ordinary indexed draws from the compute output

- Write one or more `VkDrawIndexedIndirectCommand` records on the GPU. Each record must reference the matching generated index range and vertex range, with counts from the mesh workgroup's actual outputs.
- Use a conventional vertex shader that fetches generated per-vertex data (through vertex buffers or an SSBO indexed by `gl_VertexIndex`) and forwards the original varyings to the existing fragment shader.
- Keep the original fragment shader where its interface is compatible; build the required vertex/fragment interface from the mesh program's output metadata.
- Replay with `vkCmdDrawIndexedIndirect` (or a grouped multi-draw where supported), not any `vkCmdDrawMeshTasks*` command.

### 4. Synchronization, allocation and correctness

- Add explicit compute-write to indirect-command-read and compute-write to vertex/index/shader-read barriers.
- Allocate output ranges from a bounded per-frame ring/arena. Retire them only after the frame's GPU work completes; prevent buffer reuse while submitted commands still reference it.
- Handle overflow deterministically: log and skip the affected mesh draw rather than writing past a buffer.
- Preserve primitive ordering across task workgroups and mesh workgroups, task culling, first-workgroup offsets, NVN topology, position depth conversion, clipping/culling outputs, viewport/layer outputs and primitive IDs.
- Validate with capture/test scenes that exercise task culling, large foliage/grass batches, multi-primitive meshlets, no-task mesh programs, and zero-output workgroups.

## Source areas that need coordinated changes

- `gpu/vk/vk_shader.cpp`: `mesh_program_locked`, `translate_mesh_into`, stage metadata and shader-module generation.
- `gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp` and `spirv_emit_context.*`: task/mesh epilogues, output interfaces and ISBE handling.
- `gpu/vk/vk_shader.h`: distinguish compute-emulated mesh programs from native mesh programs.
- `gpu/vk/vk_draw.cpp`: pipeline creation, descriptors, buffer binding, dispatch sequencing, indirect draw replay, resource lifetime and synchronization.
- `gpu/vk/vk_main.cpp` / `vk_internal.h`: capability selection; keep native mesh support optional, not a requirement for the fallback.
- `nx2/build.py`: include any new translator/runtime source in the Vulkan DLL build.

A likely useful first milestone is mesh-without-task (one compute dispatch per mesh draw, then one indirect indexed draw). The second milestone adds task shader payload/culling and variable child-workgroup counts. This staged route keeps the first implementation debuggable.

## Test plan for the RX 580

A meaningful test build should log an explicit renderer line such as:

```
vk: mesh fallback: compute -> indexed indirect (native mesh shaders unavailable)
```

The test report should include:
1. Whether the title/menu text is legible.
2. Whether the player, nearby geometry, doors, fences and trees render completely.
3. Whether geometry pops/disappears when moving or changing camera distance.
4. Whether loading a save completes and whether the first playable area is rendered.
5. The new log's GPU/driver line, mesh fallback counters, skipped-draw counts and any shader translation failures.

Keep a copy of the original game folder and saves before testing any experimental build.

## References

- [Vulkan mesh/task shader execution model](https://docs.vulkan.org/spec/latest/chapters/VK_NV_mesh_shader/mesh.html): task workgroups emit a variable number of mesh workgroups; mesh workgroups emit vertices and primitives.
- [VK_EXT_mesh_shader proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_mesh_shader.html): mesh shaders are compute-like workgroups and their outputs include vertex and primitive arrays.
- [Microsoft mesh-shader specification](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/MeshShader.md): describes compute-generated geometry as a traditional index-buffer/vertex-shader fallback pattern.
- Project source: `gpu/vk/vk_main.cpp`, `gpu/vk/vk_shader.cpp`, `gpu/vk/vk_draw.cpp`, and `gpu/vk/VULKAN.md`.

## Release gate

Do not publish this as an RX 580-compatible game build until:
- the compute fallback is implemented (not just documented);
- native mesh calls are absent from the fallback draw path;
- the Vulkan DLL and Windows game build compile;
- a no-mesh-shader device successfully starts and renders a scene; and
- the RX 580 tester confirms the menu, save load and main scene.

This research note is not itself a compatibility patch or binary.
