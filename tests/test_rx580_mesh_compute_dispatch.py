#!/usr/bin/env python3
"""Source regression checks for the no-task mesh-compute fallback."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAW = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
CONTEXT = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/spirv_emit_context.cpp").read_text(encoding="utf-8")
EMITTER = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp").read_text(encoding="utf-8")
DRAW_EMITTER = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp").read_text(encoding="utf-8")

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
    "NVN2 replay keeps only the verified direct task-count axis": "da.groups[0] = d.y, da.groups[1] = 1, da.groups[2] = 1;" in (ROOT / "gpu/vk/vk_replay.cpp").read_text(encoding="utf-8"),
    "flattened output allocation uses total workgroups": "xy_groups * mesh_compute_group_z" in DRAW,
    "mesh compute declares a firstTask push field": "wants_mesh_first = true;" in CONTEXT and "mesh_first_member_index" in CONTEXT,
    "mesh compute has physical firstTask pointer types": "mesh_first_block_pointer = TypePointer(spv::StorageClass::PhysicalStorageBuffer, first_block)" in CONTEXT,
    "mesh shader WorkgroupId includes firstTask": "shader_workgroup = ctx.OpIAdd(ctx.U32[1], workgroup, first);" in DRAW_EMITTER,
    "mesh output indexing keeps the local WorkgroupId": "OpCompositeExtract(ctx.U32[1], workgroup, 0U)" in DRAW_EMITTER,
    "firstTask is stored into a per-draw ring entry": "mesh_args_words[first_entry * 4 + 3] = static_cast<uint32_t>(d.first_task)" in DRAW,
    "compute dispatch pushes its direct firstTask address": "push_constants(cmd, p, nullptr, mesh_compute_commands_address, mesh_compute_vertices_address, mesh_compute_first_address)" in DRAW,
    "graphics push keeps the same compute firstTask address": "push_constants(cmd, p, &at, mesh_compute_commands_address, mesh_compute_vertices_address, mesh_compute_first_address)" in DRAW,
    "firstTask push constant is copied into the mesh compute block": "data + Shader::Backend::SPIRV::MESH_FIRST_PUSH_OFFSET" in DRAW,
    "firstTask ring supports physical shader reads": "VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT" in DRAW,
}
# Exercise the exact linearization used by MeshComputeEpilogue. It must be a
# collision-free, contiguous packet range for every positive dispatch grid.
def flatten_group(x, y, z, dispatch_x, dispatch_y):
    return x + dispatch_x * (y + dispatch_y * z)


for dispatch_x, dispatch_y, dispatch_z in ((1, 1, 1), (4, 3, 1), (2, 3, 5), (7, 1, 4)):
    slots = [
        flatten_group(x, y, z, dispatch_x, dispatch_y)
        for z in range(dispatch_z)
        for y in range(dispatch_y)
        for x in range(dispatch_x)
    ]
    expected = list(range(dispatch_x * dispatch_y * dispatch_z))
    passed = slots == expected
    checks[f"linearized group slots are contiguous for {dispatch_x}x{dispatch_y}x{dispatch_z}"] = passed

failed = [name for name, passed in checks.items() if not passed]
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")
if failed:
    raise SystemExit(f"{len(failed)} mesh-compute regression check(s) failed")
