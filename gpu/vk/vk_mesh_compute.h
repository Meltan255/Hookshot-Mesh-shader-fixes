// RX 580 mesh-shader fallback output layout.
#pragma once

#include <cstdint>
#include <limits>

namespace mkwvk {

struct MeshComputeVertex {
    float position[4];
    float generic[32][4];
};
static_assert(sizeof(MeshComputeVertex) == 33u * 4u * sizeof(float));

struct MeshComputeDraw {
    uint32_t vertex_count;
    uint32_t instance_count;
    uint32_t first_vertex;
    uint32_t first_instance;
};
static_assert(sizeof(MeshComputeDraw) == 16);

/// Word offsets in one mesh-workgroup packet.
struct MeshComputePacketLayout {
    static constexpr uint64_t kHeaderWords = 4;
    uint32_t max_vertices = 0;
    uint32_t max_primitives = 0;
    uint32_t vertex_lines = 0;
    uint32_t primitive_lines = 0;
    uint32_t corners = 3;
    constexpr uint64_t primitive_words() const { return static_cast<uint64_t>(corners) + primitive_lines; }
    constexpr uint64_t vertex_base_word() const { return kHeaderWords; }
    constexpr uint64_t primitive_base_word() const { return kHeaderWords + static_cast<uint64_t>(max_vertices) * vertex_lines; }
    constexpr uint64_t words_per_workgroup() const { return primitive_base_word() + static_cast<uint64_t>(max_primitives) * primitive_words(); }
    constexpr uint64_t bytes_per_workgroup() const { return words_per_workgroup() * sizeof(uint32_t); }
    constexpr uint64_t vertex_word(uint32_t vertex, uint32_t line) const { return vertex_base_word() + static_cast<uint64_t>(vertex) * vertex_lines + line; }
    constexpr uint64_t primitive_index_word(uint32_t primitive, uint32_t corner) const { return primitive_base_word() + static_cast<uint64_t>(primitive) * primitive_words() + corner; }
    constexpr uint64_t primitive_attribute_word(uint32_t primitive, uint32_t line) const { return primitive_base_word() + static_cast<uint64_t>(primitive) * primitive_words() + corners + line; }
    constexpr uint64_t group_base_word(uint64_t group) const { return group * words_per_workgroup(); }
    constexpr bool valid() const { return corners >= 1 && corners <= 3 && max_vertices && max_primitives && words_per_workgroup() <= std::numeric_limits<uint32_t>::max(); }
};

} // namespace mkwvk
