// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Switch 2 (NVN) task and mesh programs. They talk to the hardware through the stage buffer
// (ISBE): a task program writes the number of mesh workgroups at byte 4 of its output region and
// its payload after it; each mesh workgroup reads that region (its own index at byte 0) and
// writes its vertices and primitives to an attribute region in groups of 32, one 32-word line per
// attribute, plus the primitive count and 8-bit vertex indices to a map region. Here the task's
// output region is the task payload, the mesh output regions live in workgroup memory, and the
// epilogue (reached by every invocation, see structured_control_flow.cpp) hands them to Vulkan.

#include <algorithm>
#include <cstdlib>
#include <span>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
void Barrier(EmitContext& ctx) {
    ctx.OpControlBarrier(ctx.Const(static_cast<u32>(spv::Scope::Workgroup)),
                         ctx.Const(static_cast<u32>(spv::Scope::Workgroup)),
                         ctx.Const(static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease |
                                                    spv::MemorySemanticsMask::WorkgroupMemory)));
}

Id LocalInvocation(EmitContext& ctx) {
    return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id), 0U);
}

Id Clamped(EmitContext& ctx, Id index, u32 count) {
    return ctx.OpUMin(ctx.U32[1], index, ctx.Const(count - 1));
}

Id PayloadWord(EmitContext& ctx, Id index) {
    return ctx.OpAccessChain(ctx.mesh_payload_u32, ctx.mesh_payload,
                             Clamped(ctx, index, ctx.mesh_info.payload_words));
}

Id SharedWord(EmitContext& ctx, Id array, Id index) {
    return ctx.OpAccessChain(ctx.mesh_shared_u32, array, index);
}

// Runs body under `if (condition)`.
template <typename Func>
void If(EmitContext& ctx, Id condition, Func&& body) {
    const Id then_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, then_label, merge_label);
    ctx.AddLabel(then_label);
    body();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

// Copies the lines of element `index` (a vertex or a primitive) from the attribute region.
void CopyLines(EmitContext& ctx, Id index, u32 base, u32 num_lines,
               const std::vector<EmitContext::MeshLineTarget>& targets) {
    // Word of line 0: base + (index / 32) * num_lines * 32 + index % 32.
    const Id group{ctx.OpShiftRightLogical(ctx.U32[1], index, ctx.Const(5U))};
    const Id slot{ctx.OpBitwiseAnd(ctx.U32[1], index, ctx.Const(31U))};
    const Id first{ctx.OpIAdd(ctx.U32[1],
                              ctx.OpIMul(ctx.U32[1], group, ctx.Const(num_lines * 32)),
                              ctx.OpIAdd(ctx.U32[1], slot, ctx.Const(base)))};
    for (u32 line = 0; line < num_lines; ++line) {
        const EmitContext::MeshLineTarget& target{targets[line]};
        if (!Sirit::ValidId(target.variable)) {
            continue;
        }
        const Id word{ctx.OpLoad(
            ctx.U32[1],
            SharedWord(ctx, ctx.mesh_attributes, ctx.OpIAdd(ctx.U32[1], first, ctx.Const(line * 32))))};
        const Id value{target.is_float ? ctx.OpBitcast(ctx.F32[1], word) : word};
        const Id pointer_type{target.is_float ? ctx.output_f32 : ctx.output_u32};
        const Id pointer{target.element == ~0U
                             ? ctx.OpAccessChain(pointer_type, target.variable, index)
                             : ctx.OpAccessChain(pointer_type, target.variable, index,
                                                 ctx.Const(target.element))};
        ctx.OpStore(pointer, value);
    }
}

// OpenGL-style clip space: z = (z + w) / 2, from the stage buffer's position lines.
void ConvertDepth(EmitContext& ctx, Id vertex) {
    const auto& lines{ctx.mesh_info.vertex_lines};
    const auto z_line{std::ranges::find(lines, static_cast<u32>(IR::Attribute::PositionZ))};
    const auto w_line{std::ranges::find(lines, static_cast<u32>(IR::Attribute::PositionW))};
    if (z_line == lines.end() || w_line == lines.end()) {
        return;
    }
    const u32 num_lines{static_cast<u32>(lines.size())};
    const Id group{ctx.OpShiftRightLogical(ctx.U32[1], vertex, ctx.Const(5U))};
    const Id slot{ctx.OpBitwiseAnd(ctx.U32[1], vertex, ctx.Const(31U))};
    const Id first{ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], group, ctx.Const(num_lines * 32)),
                              slot)};
    const auto read = [&](auto line) {
        const u32 offset{static_cast<u32>(line - lines.begin()) * 32};
        return ctx.OpBitcast(ctx.F32[1],
                             ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_attributes,
                                                               ctx.OpIAdd(ctx.U32[1], first,
                                                                          ctx.Const(offset)))));
    };
    const Id z{ctx.OpFMul(ctx.F32[1], ctx.OpFAdd(ctx.F32[1], read(z_line), read(w_line)),
                          ctx.Const(0.5f))};
    ctx.OpStore(ctx.OpAccessChain(ctx.output_f32, ctx.output_position, vertex, ctx.Const(2U)), z);
}

u32 task_cap(u32 threads) {   // (RX_VK_TASK_CAP=<n>: mesh workgroups a task may start; 0: one a thread)
    static const char* e = getenv("RX_VK_TASK_CAP");
    const u32 n = e ? static_cast<u32>(atoi(e)) : 0U;
    return n ? n : threads;
}

void TaskEpilogue(EmitContext& ctx) {
    Barrier(ctx);
    // (hookshot) at most one mesh workgroup a task thread: each thread keeps or culls one meshlet (the programs compact them by subgroup
    // ballot). It was 65535, and a payload word a task left unwritten is anything: when the task culled everything, tens of thousands of
    // mesh workgroups for a draw held the GPU past Windows' two seconds, and the device was lost, on about half the first starts
    const Id count{ctx.OpUMin(ctx.U32[1], ctx.OpLoad(ctx.U32[1], PayloadWord(ctx, ctx.Const(1U))),
                              ctx.Const(task_cap(ctx.mesh_threads)))};
    const Id one{ctx.Const(1U)};
    // Marker for OpEmitMeshTasksEXT count, 1, 1, payload (PatchMeshInstructions).
    ctx.OpStore(count, one);
    ctx.OpStore(one, ctx.mesh_payload);
    ctx.mesh_tasks_marker = {count, one, one, ctx.mesh_payload};
}

void MeshEpilogue(EmitContext& ctx) {
    const MeshInfo& mesh{ctx.mesh_info};
    Barrier(ctx);
    const Id count{ctx.OpUMin(ctx.U32[1], ctx.OpLoad(ctx.U32[1], ctx.mesh_primitive_count),
                              ctx.Const(mesh.max_primitives))};
    const Id max_vertices{ctx.Const(mesh.max_vertices)};
    // Marker for OpSetMeshOutputsEXT max_vertices, count (PatchMeshInstructions).
    ctx.OpStore(max_vertices, count);
    ctx.mesh_outputs_marker = {max_vertices, count};

    const Id invocation{LocalInvocation(ctx)};
    const u32 threads{ctx.mesh_threads};
    const bool convert_depth{ctx.runtime_info.convert_depth_mode && !ctx.profile.support_native_ndc &&
                             Sirit::ValidId(ctx.output_position)};
    const u32 vertex_lines{static_cast<u32>(mesh.vertex_lines.size())};
    for (u32 first = 0; first < mesh.max_vertices; first += threads) {
        const Id vertex{ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first))};
        const auto copy = [&] {
            CopyLines(ctx, vertex, 0, vertex_lines, ctx.mesh_vertex_targets);
            if (convert_depth) {
                ConvertDepth(ctx, vertex);
            }
        };
        if (first + threads > mesh.max_vertices) {
            If(ctx, ctx.OpULessThan(ctx.U1, vertex, max_vertices), copy);
        } else {
            copy();
        }
    }
    const u32 primitive_lines{static_cast<u32>(mesh.primitive_lines.size())};
    for (u32 first = 0; first < mesh.max_primitives; first += threads) {
        const Id primitive{ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first))};
        If(ctx, ctx.OpULessThan(ctx.U1, primitive, count), [&] {
            // Vertex indices: map bytes 4 + 3 * primitive (lines: 4 + 2 * primitive).
            const u32 corners{mesh.topology == 1 ? 2U : 3U};
            const Id byte{ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], primitive, ctx.Const(corners)),
                                     ctx.Const(4U))};
            std::array<Id, 3> indices;
            for (u32 element = 0; element < corners; ++element) {
                const Id address{ctx.OpIAdd(ctx.U32[1], byte, ctx.Const(element))};
                // (hookshot) within the vertices: an index byte the program left unwritten is anything, and an index
                // past the vertex count is undefined (a device fault on NVIDIA)
                indices[element] = Clamped(ctx, ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_map, address)),
                                           mesh.max_vertices);
            }
            const Id output_indices{ctx.TypePointer(spv::StorageClass::Output, ctx.U32[corners])};
            ctx.OpStore(ctx.OpAccessChain(output_indices, ctx.mesh_primitive_indices, primitive),
                        ctx.OpCompositeConstruct(ctx.U32[corners], std::span<const Id>(indices.data(), corners)));
            CopyLines(ctx, primitive, ctx.mesh_primitive_base, primitive_lines,
                      ctx.mesh_primitive_targets);
        });
    }
}
} // Anonymous namespace

Id EmitIsbeRead(EmitContext& ctx, Id address) {
    const Id workgroup{
        ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.workgroup_id), 0U)};
    if (ctx.stage == Stage::Task) {
        if (ctx.wants_mesh_first && Sirit::ValidId(ctx.rescaling_push_constants)) {
            // (hookshot) + the draw's first workgroup: NVN draws [first, first + count). Without it every draw began at 0: the
            // foliage's draws (first up to ~1900) drew the first records again and again, and the forest had no leaves
            const Id table{ctx.OpLoad(ctx.U64, ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U64),
                                                                  ctx.rescaling_push_constants, ctx.Const(ctx.mesh_first_member_index)))};
            const Id draw{ctx.OpLoad(ctx.U32[1], ctx.draw_index)};
            const Id at{ctx.OpIAdd(ctx.U64, table, ctx.OpUConvert(ctx.U64, ctx.OpIMul(ctx.U32[1], draw, ctx.Const(16U))))};
            const Id member{ctx.OpAccessChain(ctx.mesh_first_member_pointer, ctx.OpConvertUToPtr(ctx.mesh_first_block_pointer, at),
                                              ctx.Const(0U))};
            const Id first{ctx.OpLoad(ctx.U32[1], member, spv::MemoryAccessMask::Aligned, 4U)};
            return ctx.OpIAdd(ctx.U32[1], workgroup, first);
        }
        return workgroup; // the input region holds the workgroup index (VERTEX_ID)
    }
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], address, ctx.Const(2U))};
    const Id word{ctx.OpLoad(ctx.U32[1], PayloadWord(ctx, index))};
    return ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, address, ctx.u32_zero_value), workgroup,
                        word);
}

void EmitIsbeWriteAttribute(EmitContext& ctx, Id address, Id value) {
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], address, ctx.Const(2U))};
    if (ctx.stage == Stage::Task) {
        ctx.OpStore(PayloadWord(ctx, index), value);
        return;
    }
    ctx.OpStore(SharedWord(ctx, ctx.mesh_attributes, Clamped(ctx, index, ctx.mesh_attribute_words)),
                value);
}

void EmitIsbeWriteMapU8(EmitContext& ctx, Id address, Id value) {
    if (ctx.stage != Stage::Mesh) {
        throw NotImplementedException("Stage buffer map write outside mesh programs");
    }
    ctx.OpStore(SharedWord(ctx, ctx.mesh_map, Clamped(ctx, address, ctx.mesh_map_bytes)),
                ctx.OpBitwiseAnd(ctx.U32[1], value, ctx.Const(0xffU)));
}

void EmitIsbeWriteMapU32(EmitContext& ctx, const IR::Value& address, Id value) {
    // The primitive count: the map is written from its end, so its first word is at byte 3.
    if (ctx.stage != Stage::Mesh || !address.IsImmediate() || address.U32() != 3) {
        throw NotImplementedException("32-bit stage buffer map write");
    }
    ctx.OpStore(ctx.mesh_primitive_count, value);
}

void EmitMeshPrologue(EmitContext& ctx) {
    if (ctx.stage == Stage::Task) {
        // (hookshot) the count starts at nothing: task payload memory holds whatever the last task left, and a task that keeps no meshlet
        // never writes it. The barrier: every thread has done this before any of them goes on to write the real count
        ctx.OpStore(PayloadWord(ctx, ctx.Const(1U)), ctx.u32_zero_value);
        Barrier(ctx);
    }
    if (ctx.stage != Stage::Mesh) {
        return;
    }
    // Every invocation stores the zero: no branch here, the entry block's label is a phi
    // predecessor for later blocks.
    ctx.OpStore(ctx.mesh_primitive_count, ctx.u32_zero_value);
    Barrier(ctx);
}

namespace {
Id MeshComputeOutputWord(EmitContext& ctx, Id base, Id word_index) {
    const Id block = ctx.OpConvertUToPtr(ctx.mesh_compute_output_block_pointer, base);
    return ctx.OpAccessChain(ctx.mesh_compute_output_word_pointer, block, ctx.Const(0U), word_index);
}
Id MeshAttributeWord(EmitContext& ctx, Id element, u32 base, u32 line, u32 line_count) {
    const Id group = ctx.OpShiftRightLogical(ctx.U32[1], element, ctx.Const(5U));
    const Id slot = ctx.OpBitwiseAnd(ctx.U32[1], element, ctx.Const(31U));
    const Id first = ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], group, ctx.Const(line_count * 32U)),
                                ctx.OpIAdd(ctx.U32[1], slot, ctx.Const(base + line * 32U)));
    return ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_attributes, first));
}
Id MeshMapVertex(EmitContext& ctx, Id primitive, u32 corner, u32 corners) {
    const Id address = ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], primitive, ctx.Const(corners)),
                                  ctx.Const(4U + corner));
    return Clamped(ctx, ctx.OpLoad(ctx.U32[1], SharedWord(ctx, ctx.mesh_map, address)),
                   ctx.mesh_info.max_vertices);
}
void MeshComputeEpilogue(EmitContext& ctx) {
    const MeshInfo& mesh = ctx.mesh_info;
    Barrier(ctx);
    const Id count = ctx.OpUMin(ctx.U32[1], ctx.OpLoad(ctx.U32[1], ctx.mesh_primitive_count),
                                ctx.Const(mesh.max_primitives));
    const Id group = ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.workgroup_id), 0U);
    const Id group64 = ctx.OpUConvert(ctx.U64, group);
    const Id commands = ctx.OpLoad(ctx.U64,
        ctx.OpAccessChain(ctx.mesh_compute_push_u64, ctx.rescaling_push_constants,
                          ctx.Const(ctx.mesh_compute_commands_member_index)));
    const Id vertices = ctx.OpLoad(ctx.U64,
        ctx.OpAccessChain(ctx.mesh_compute_push_u64, ctx.rescaling_push_constants,
                          ctx.Const(ctx.mesh_compute_vertices_member_index)));
    const u32 corners = mesh.topology == 1 ? 2U : 3U;
    const u32 vertex_lines = static_cast<u32>(mesh.vertex_lines.size());
    const u32 primitive_lines = static_cast<u32>(mesh.primitive_lines.size());
    const bool convert_depth = ctx.runtime_info.convert_depth_mode && !ctx.profile.support_native_ndc;
    const auto line_index = [](const std::vector<u32>& lines, IR::Attribute attribute) -> int {
        const auto it = std::ranges::find(lines, static_cast<u32>(attribute));
        return it == lines.end() ? -1 : static_cast<int>(it - lines.begin());
    };
    const int pz = line_index(mesh.vertex_lines, IR::Attribute::PositionZ);
    const int pw = line_index(mesh.vertex_lines, IR::Attribute::PositionW);
    const Id invocation = LocalInvocation(ctx);
    for (u32 first = 0; first < mesh.max_primitives; first += ctx.mesh_threads) {
        const Id primitive = ctx.OpIAdd(ctx.U32[1], invocation, ctx.Const(first));
        If(ctx, ctx.OpULessThan(ctx.U1, primitive, count), [&] {
            for (u32 corner = 0; corner < corners; ++corner) {
                const Id vertex = MeshMapVertex(ctx, primitive, corner, corners);
                const Id record = ctx.OpIAdd(ctx.U32[1],
                    ctx.OpIMul(ctx.U32[1], primitive, ctx.Const(corners)), ctx.Const(corner));
                const Id global_record = ctx.OpIAdd(ctx.U32[1],
                    ctx.OpIMul(ctx.U32[1], group, ctx.Const(mesh.max_primitives * corners)), record);
                const Id record64 = ctx.OpUConvert(ctx.U64, global_record);
                const Id word_base = ctx.OpIMul(ctx.U64, record64, ctx.Constant(ctx.U64, 33U * 4U));
                const auto store_word = [&](u32 slot, Id value) {
                    const Id word = ctx.OpIAdd(ctx.U64, word_base, ctx.Constant(ctx.U64, slot));
                    const Id byte_offset = ctx.OpIMul(ctx.U64, word, ctx.Constant(ctx.U64, 4U));
                    const Id addr = ctx.OpIAdd(ctx.U64, vertices, byte_offset);
                    ctx.OpStore(MeshComputeOutputWord(ctx, addr, ctx.Const(0U)), value);
                };
                Id z_value = ctx.u32_zero_value, w_value = ctx.Const(0x3f800000U);
                if (pz >= 0) z_value = MeshAttributeWord(ctx, vertex, 0, static_cast<u32>(pz), vertex_lines);
                if (pw >= 0) w_value = MeshAttributeWord(ctx, vertex, 0, static_cast<u32>(pw), vertex_lines);
                for (u32 line = 0; line < vertex_lines; ++line) {
                    const auto attribute = static_cast<IR::Attribute>(mesh.vertex_lines[line]);
                    u32 slot = 0;
                    bool write = true;
                    if (attribute >= IR::Attribute::PositionX && attribute <= IR::Attribute::PositionW) {
                        slot = static_cast<u32>(attribute) - static_cast<u32>(IR::Attribute::PositionX);
                        if (convert_depth && attribute == IR::Attribute::PositionZ) {
                            const Id z = ctx.OpBitcast(ctx.F32[1], z_value);
                            const Id w = ctx.OpBitcast(ctx.F32[1], w_value);
                            const Id converted = ctx.OpFMul(ctx.F32[1], ctx.OpFAdd(ctx.F32[1], z, w),
                                                           ctx.Const(0.5f));
                            store_word(slot, ctx.OpBitcast(ctx.U32[1], converted));
                            write = false;
                        }
                    } else if (IR::IsGeneric(attribute)) {
                        slot = 4U + IR::GenericAttributeIndex(attribute) * 4U +
                               IR::GenericAttributeElement(attribute);
                    } else {
                        write = false;
                    }
                    if (write) store_word(slot, MeshAttributeWord(ctx, vertex, 0, line, vertex_lines));
                }
                for (u32 line = 0; line < primitive_lines; ++line) {
                    const auto attribute = static_cast<IR::Attribute>(mesh.primitive_lines[line]);
                    if (IR::IsGeneric(attribute)) {
                        const u32 slot = 4U + IR::GenericAttributeIndex(attribute) * 4U +
                                         IR::GenericAttributeElement(attribute);
                        store_word(slot, MeshAttributeWord(ctx, primitive, ctx.mesh_primitive_base,
                                                           line, primitive_lines));
                    }
                }
            }
        });
    }
    Barrier(ctx);
    If(ctx, ctx.OpIEqual(ctx.U1, invocation, ctx.u32_zero_value), [&] {
        const Id cmd_base = ctx.OpIMul(ctx.U64, group64, ctx.Constant(ctx.U64, 16U));
        const Id vertex_count = ctx.OpIMul(ctx.U32[1], count, ctx.Const(corners));
        const Id first_vertex = ctx.OpIMul(ctx.U32[1], group, ctx.Const(mesh.max_primitives * corners));
        const Id values[4] = {vertex_count, ctx.Const(1U), first_vertex, ctx.u32_zero_value};
        for (u32 i = 0; i < 4; ++i) {
            const Id byte_offset = ctx.OpIAdd(ctx.U64, cmd_base, ctx.Constant(ctx.U64, i * 4U));
            const Id addr = ctx.OpIAdd(ctx.U64, commands, byte_offset);
            ctx.OpStore(MeshComputeOutputWord(ctx, addr, ctx.Const(0U)), values[i]);
        }
    });
}
} // namespace

void EmitMeshEpilogue(EmitContext& ctx) {
    if (ctx.stage == Stage::Task) {
        TaskEpilogue(ctx);
    } else if (ctx.stage == Stage::Mesh) {
        if (ctx.runtime_info.mesh_compute_emulation) MeshComputeEpilogue(ctx);
        else MeshEpilogue(ctx);
    }
}

} // namespace Shader::Backend::SPIRV
