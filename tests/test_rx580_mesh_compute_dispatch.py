#!/usr/bin/env python3
"""Source regression checks for the no-task 3D mesh-compute fallback."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAW = (ROOT / "gpu/vk/vk_draw.cpp").read_text(encoding="utf-8")
REPLAY = (ROOT / "gpu/vk/vk_replay.cpp").read_text(encoding="utf-8")
CONTEXT = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/spirv_emit_context.cpp").read_text(encoding="utf-8")
EMITTER = (ROOT / "gpu/vk/eden/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp").read_text(encoding="utf-8")

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
    "mesh draw replay preserves both direct workgroup counts": "da.groups[0] = d.y, da.groups[1] = d.z, da.groups[2] = 1;" in REPLAY,
    "flattened output allocation uses total workgroups": "xy_groups * mesh_compute_group_z" in DRAW,
}
failed = [name for name, passed in checks.items() if not passed]
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")
if failed:
    raise SystemExit(f"{len(failed)} mesh-compute regression check(s) failed")
