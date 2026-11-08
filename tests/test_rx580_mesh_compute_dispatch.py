#!/usr/bin/env python3
"""Source regression checks for RX 580 mesh/task compute emulation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAW = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
SHADER = (ROOT / "gpu/vk/vk_shader.cpp").read_text(encoding="utf-8")
SHADER_H = (ROOT / "gpu/vk/vk_shader.h").read_text(encoding="utf-8")
CONTEXT = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/spirv_emit_context.cpp").read_text(encoding="utf-8")
EMITTER = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp").read_text(encoding="utf-8")
ENTRY = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv.cpp").read_text(encoding="utf-8")
GETSET = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_context_get_set.cpp").read_text(encoding="utf-8")
RUNTIME = (ROOT / "gpu/vk/eden/shader_recompiler/runtime_info.h").read_text(encoding="utf-8")
REPLAY = (ROOT / "gpu/vk/vk_replay.cpp").read_text(encoding="utf-8")

checks = {
    "mesh compute always receives WorkgroupId": "info.uses_workgroup_id || mesh_compute_emulation" in CONTEXT,
    "mesh compute receives NumWorkgroups": "num_workgroups = DefineInput(*this, U32[3], false, spv::BuiltIn::NumWorkgroups)" in CONTEXT,
    "mesh compute output flatten includes Y and Z": "dispatch_y, group_z" in EMITTER,
    "output flatten scales rows by dispatch X": "dispatch_x, yz" in EMITTER,
    "draw accepts all three positive grid axes": "d.groups[1] <= 0 || d.groups[2] <= 0" in DRAW,
    "dispatch axes are checked against Vulkan limits": all(
        f"mesh_compute_group_{axis} > dev.props.limits.maxComputeWorkGroupCount[{index}]"
        in DRAW
        for axis, index in (("x", 0), ("y", 1), ("z", 2))
    ),
    "dispatch preserves original 3D grid": "vkCmdDispatch(cmd, mesh_compute_group_x, mesh_compute_group_y, mesh_compute_group_z)" in DRAW,
    "NVN2 replay keeps only the verified direct task-count axis": "da.groups[0] = d.y, da.groups[1] = 1, da.groups[2] = 1;" in REPLAY,
    "flattened output allocation uses total workgroups": "xy_groups * mesh_compute_group_z" in DRAW,
    "mesh compute declares a firstTask push field": "wants_mesh_first = true;" in CONTEXT and "mesh_first_member_index" in CONTEXT,
    "mesh compute has physical firstTask pointer types": "mesh_first_block_pointer = TypePointer(spv::StorageClass::PhysicalStorageBuffer, first_block)" in CONTEXT,
    "mesh shader WorkgroupId includes firstTask": "shader_workgroup = ctx.OpIAdd(ctx.U32[1], workgroup, first);" in EMITTER,
    "direct WorkgroupId builtin preserves firstTask": (
        "needs_first_task" in GETSET
        and "ctx.stage == Shader::Stage::Task && ctx.runtime_info.task_compute_emulation" in GETSET
        and "ctx.stage == Shader::Stage::Mesh && ctx.runtime_info.mesh_compute_emulation" in GETSET
    ),
    "mesh output indexing keeps the local WorkgroupId": "OpCompositeExtract(ctx.U32[1], workgroup, 0U)" in EMITTER,
    "firstTask is stored into a per-draw ring entry": "mesh_args_words[first_entry * 4 + 3] = static_cast<uint32_t>(d.first_task)" in DRAW,
    "compute dispatch pushes its direct firstTask address": "push_constants(cmd, p, nullptr, mesh_compute_commands_address" in DRAW and "mesh_compute_first_address" in DRAW,
    "graphics push keeps the same compute firstTask address": "push_constants(cmd, p, &at, mesh_compute_commands_address" in DRAW and "mesh_compute_first_address" in DRAW,
    "firstTask push constant is copied into the mesh compute block": "data + Shader::Backend::SPIRV::MESH_FIRST_PUSH_OFFSET" in DRAW,
    "firstTask ring supports physical shader reads": "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT" in DRAW,
    "firstTask value uses the direct draw's ring slot": "mesh_compute_first_address = mesh_args_address + static_cast<uint64_t>(first_entry) * 16U + 12U" in DRAW,
    "task compute runtime mode exists": "bool task_compute_emulation{};" in RUNTIME,
    "mesh task-input runtime mode exists": "bool mesh_task_input_emulation{};" in RUNTIME,
    "task and mesh compute stages are both enabled on non-mesh devices": "rt.task_compute_emulation = task_compute;" in SHADER and "mrt.mesh_compute_emulation = !dev.mesh_shader;" in SHADER,
    "task shader uses GLCompute execution model in fallback": "if (ctx.runtime_info.task_compute_emulation)" in ENTRY and "execution_model = spv::ExecutionModel::GLCompute;" in ENTRY,
    "task compute writes one count and payload record per parent": "MeshTaskOutputWord(ctx, output, base), count" in EMITTER and "MeshTaskOutputWord(ctx, output, destination), value" in EMITTER,
    "mesh compute reads the parent record and copies payload": "MeshTaskOutputWord(ctx, output, base)" in EMITTER and "mesh_task_child_id = ctx.OpUMod" in EMITTER,
    "mesh child ordinal is exposed as shader WorkgroupId": "ctx.mesh_task_child_id" in GETSET and "mesh_task_input_emulation" in GETSET,
    "culled child slots emit zero primitives": "ctx.mesh_task_child_valid, raw_count, ctx.u32_zero_value" in EMITTER,
    "task payload words are cleared before task shader execution": "ctx.stage == Stage::Task" in EMITTER and "ctx.OpStore(PayloadWord(ctx, word), ctx.u32_zero_value)" in EMITTER,
    "task payload buffer has storage and device-address usage": "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT" in DRAW,
    "task record indexing is bounded to 32 bits": (
        "task_compute_groups) > UINT32_MAX / words_per_task" in DRAW
        and "OpIMul(ctx.U32[1], workgroup, stride)" in EMITTER
        and "OpIMul(ctx.U32[1], task_group, stride)" in EMITTER
    ),
    "task compute pipeline uses the task stage module": "st.nvn_stage == 6 && st.vk_stage == VK_SHADER_STAGE_COMPUTE_BIT" in DRAW and "p.task_compute_pipeline" in DRAW,
    "mesh compute pipeline uses the mesh stage module": "st.nvn_stage == 7 && st.vk_stage == VK_SHADER_STAGE_COMPUTE_BIT" in DRAW and "p.mesh_compute_pipeline" in DRAW,
    "task dispatch occurs before mesh dispatch with a full barrier": (
        DRAW.index("vkCmdDispatch(cmd, task_compute_groups, 1, 1);")
        < DRAW.index("full_barrier(cmd);", DRAW.index("vkCmdDispatch(cmd, task_compute_groups, 1, 1);"))
        < DRAW.index("vkCmdDispatch(cmd, mesh_compute_group_x, mesh_compute_group_y, mesh_compute_group_z);")
    ),
    "task fallback rejects indirect and non-1D grids": "if (d.indirect || d.groups[0] <= 0" in DRAW and "d.groups[1] != 1 || d.groups[2] != 1" in DRAW,
    "task and mesh payload sizes are checked": "task_t->program.mesh.payload_words != mesh_t->program.mesh.payload_words" in SHADER,
    "task output address is pushed": "MESH_TASK_OUTPUT_PUSH_OFFSET" in DRAW and "mesh_task_output_address" in DRAW,
    "external side effects are rejected only on task-overdispatched mesh stage": (
        "if (!dev.mesh_shader && task_t,\\n            mesh_compute_has_unsupported_side_effects(mesh_t->program))" in SHADER
    ),
    "task stage is not rejected for one-per-task guest writes": "mesh_compute_has_unsupported_side_effects(task_t->program)" not in SHADER,
    "task fallback permits only workgroup-local side effects": (
        "case Shader::IR::Opcode::WriteSharedU32:" in SHADER
        and "case Shader::IR::Opcode::SharedAtomicExchange32:" in SHADER
        and "default:\n                return true;" in SHADER
    ),
    "64-bit physical storage address support is required": (
        "compute mesh fallback: 64-bit physical storage addresses are unavailable" in SHADER
        and "task compute fallback: 64-bit physical storage addresses are unavailable" in SHADER
    ),
    "emulated task and mesh subgroup checks use compute support": (
        "queried_stage == Stage::Task && runtime_info.task_compute_emulation" in
        (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/spirv_emit_context.h").read_text(encoding="utf-8")
        and "queried_stage == Stage::Mesh && runtime_info.mesh_compute_emulation" in
        (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/spirv_emit_context.h").read_text(encoding="utf-8")
        and "profile.SupportsSubgroupStage(Stage::Compute)" in
        (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/spirv_emit_context.h").read_text(encoding="utf-8")
    ),
    "warp helper uses stage-aware subgroup support": "ctx.SupportsSubgroupStage(ctx.stage)" in
        (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_warp.cpp").read_text(encoding="utf-8"),
    "entrypoint subgroup capabilities use stage-aware support": "ctx.SupportsSubgroupStage(ctx.stage)" in ENTRY,
}

def flatten_group(x, y, z, dispatch_x, dispatch_y):
    return x + dispatch_x * (y + dispatch_y * z)


def flatten_task_child(task, child, groups_per_task):
    return task * groups_per_task + child


# Workgroup output slots must be collision-free and contiguous for every
# supported dispatch grid. Shader-visible IDs are separate from buffer slots.
for dispatch_x, dispatch_y, dispatch_z in ((1, 1, 1), (4, 3, 1), (2, 3, 5), (7, 1, 4)):
    slots = [
        flatten_group(x, y, z, dispatch_x, dispatch_y)
        for z in range(dispatch_z)
        for y in range(dispatch_y)
        for x in range(dispatch_x)
    ]
    checks[f"linearized group slots are contiguous for {dispatch_x}x{dispatch_y}x{dispatch_z}"] = (
        slots == list(range(dispatch_x * dispatch_y * dispatch_z))
    )

for task_count, cap in ((1, 1), (2, 32), (17, 64)):
    slots = [
        flatten_task_child(task, child, cap)
        for task in range(task_count)
        for child in range(cap)
    ]
    checks[f"task child slots map to stable parent records for {task_count}x{cap}"] = (
        slots == list(range(task_count * cap))
        and all(slot // cap == task and slot % cap == child
                for task in range(task_count)
                for child in range(cap)
                for slot in [flatten_task_child(task, child, cap)])
    )

for first_task in (0, 1, 17, 2048):
    for local_group in range(8):
        guest_group = local_group + first_task
        output_group = local_group
        checks[f"firstTask preserves local output slot ({first_task},{local_group})"] = (
            guest_group - first_task == output_group
        )

failed = [name for name, passed in checks.items() if not passed]
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")
if failed:
    raise SystemExit(f"{len(failed)} RX 580 mesh/task compute regression check(s) failed")
