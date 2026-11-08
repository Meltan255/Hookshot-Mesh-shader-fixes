# RX 580 compatibility: no-task mesh shaders via compute + indirect draws

Status: **partial research-branch implementation; compiled by CI, not yet validated on an RX 580.** The current code translates only direct one-dimensional mesh draws with no task stage into a compute dispatch plus generated vertex data and indirect conventional draws. It is not a complete task/mesh-shader replacement.

This note records the RX 580 failure mode and a concrete fallback design. It deliberately does not claim that a capability check or an environment-variable change can make mesh shaders work on hardware that does not expose `VK_EXT_mesh_shader`.

## Evidence from the supplied RX 580 log

The user's log reports:

```
AMD Radeon RX 580 2048SP, Vulkan 1.3, driver 0x800117;
mesh shaders no, host memory import yes, pipeline libraries no
```

The Vulkan renderer starts successfully, so this is not a Vulkan-loader failure. The missing capability is specifically mesh/task shader execution.

In `gpu/vk/vk_main.cpp`, native mesh support is enabled only when `VK_EXT_mesh_shader` and both `meshShader` and `taskShader` features are present. On a device without those features, `mesh_program_locked()` now rejects task-stage mesh programs but allows the no-task path to attempt compute emulation. The draw path dispatches the translated mesh stage as compute, writes a conventional vertex buffer and one indirect draw command per workgroup, then uses a generated vertex shader with the original fragment shader. Unsupported draws are deliberately skipped rather than sent to native mesh commands.

The WebGPU backend is not an equivalent workaround: `gpu/vk/VULKAN.md` explicitly says the WebGPU renderer has no mesh stage and drops those draws, which produces missing high-detail geometry / LOD pop-in.

## Separate issue in the same log: cooperative-matrix compute programs

The log also contains ten failed compute programs reporting:

```
not translated: a network left out on this card
rejected compute ...
```

This message is emitted by the cooperative-matrix policy in `gpu/vk/vk_shader.cpp`. The surrounding source comment says this policy skips most of the game's cooperative-matrix movie-upscaler networks on non-NVIDIA GPUs because their software matrix multiply path was prohibitively slow; two explicitly whitelisted networks are retained, including one associated with the dynamic-resolution scene.

This is a separate compute-shader policy, not proof of a mesh-shader failure. It should not be disabled wholesale as a mesh workaround. The supplied log alone does not establish that these ten failures explain the broken menu text; the menu/font issue needs its own capture after a real mesh fallback exists.

## Fallback architecture and remaining work

The current branch emulates only eligible no-task mesh programs using ordinary compute and graphics stages. It does not rename a mesh execution model to a vertex execution model: it translates the mesh workgroup into compute writes, then consumes those writes through a conventional vertex-input pipeline. Task payloads and task-stage dispatch remain future work.

### 1. Task stage remains unsupported

The current code does not translate task stages. `mesh_program_locked()` rejects any mesh program that has a task stage when native mesh shaders are unavailable. A future implementation must translate task-stage IR to compute, preserve ISBE payloads and barriers, and feed its emitted workgroups into mesh compute without reading GPU results back to the CPU.

### 2. No-task mesh stage is partially implemented

The current branch translates eligible no-task mesh stages to SPIR-V compute and writes vertex and indirect-command records through device addresses. It preserves shared mesh output and per-primitive generic values. The acceptance checks cover line-list and triangle-list output with position X/Y/Z and generic varyings. Task payloads, indirect mesh dispatches and multi-dimensional dispatch grids are not supported. Hardware/runtime validation is still outstanding.

### 3. Conventional indirect draws

The current code expands each output primitive to a line-list or triangle-list vertex sequence, so it does not need to build an index buffer. The compute stage writes a `VkDrawIndirectCommand` per workgroup; a generated vertex shader reads the output vertex buffer and forwards generic varyings to the original fragment shader. When `multiDrawIndirect` is unavailable, the renderer issues one indirect draw command at a time.

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

The first no-task milestone now exists in the research branch and passes the Windows renderer compile check. The next milestone is task-stage payload/culling translation, followed by broader topology, indirect-dispatch and multidimensional-workgroup coverage. None of those milestones has been confirmed on RX 580 hardware.

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

Do not describe this as generally RX 580-compatible until:
- the supported no-task path passes a real render test on a device without mesh-shader support;
- native mesh calls are absent from the fallback draw path;
- the Vulkan DLL and Windows game build compile;
- a no-mesh-shader device successfully starts and renders a scene; and
- the RX 580 tester confirms the menu, save load and main scene.

This research note is not itself a compatibility patch or binary.
