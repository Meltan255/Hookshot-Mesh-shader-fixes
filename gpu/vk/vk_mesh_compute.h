// RX 580 mesh-shader fallback: GPU packet format shared by mesh compute and draw replay.
// A compute workgroup writes one fixed-size packet. A later flatten pass consumes these
// packets to produce conventional vertex data and draw-indirect commands.
#pragma once

#include <cstdint>
#include <limits>

namespace mkwvk {

struct MeshComputePacketHeader {
    uint32_t vertex_count;
    uint32_t primitive_count;
    uint32_t topology;
    uint32_t vertex_line_count;
};
static_assert(sizeof(MeshComputePacketHeader) == 4 * sizeof(uint32_t));

/// Word offsets in the per-workgroup compute output packet.
/// Vertex attributes are interleaved by vertex. Each primitive stores its corner indices
/// followed by its per-primitive attribute words.
struct MeshComputePacketLayout {
    static constexpr uint64_t kHeaderWords = 4;

    uint32_t max_vertices = 0;
    uint32_t max_primitives = 0;
    uint32_t vertex_lines = 0;
    uint32_t primitive_lines = 0;
    uint32_t corners = 3;

    constexpr uint64_t primitive_words() const {
        return static_cast<uint64_t>(corners) + primitive_lines;
    }
    constexpr uint64_t vertex_base_word() const {
        return kHeaderWords;
    }
    constexpr uint64_t primitive_base_word() const {
        return kHeaderWords + static_cast<uint64_t>(max_vertices) * vertex_lines;
    }
    constexpr uint64_t words_per_workgroup() const {
        return primitive_base_word() + static_cast<uint64_t>(max_primitives) * primitive_words();
    }
    constexpr uint64_t bytes_per_workgroup() const {
        return words_per_workgroup() * sizeof(uint32_t);
    }
    constexpr uint64_t vertex_word(uint32_t vertex, uint32_t line) const {
        return vertex_base_word() + static_cast<uint64_t>(vertex) * vertex_lines + line;
    }
    constexpr uint64_t primitive_index_word(uint32_t primitive, uint32_t corner) const {
        return primitive_base_word() + static_cast<uint64_t>(primitive) * primitive_words() + corner;
    }
    constexpr uint64_t primitive_attribute_word(uint32_t primitive, uint32_t line) const {
        return primitive_base_word() + static_cast<uint64_t>(primitive) * primitive_words() + corners + line;
    }
    constexpr uint64_t group_base_word(uint64_t group) const {
        return group * words_per_workgroup();
    }
    constexpr bool valid() const {
        return corners >= 1 && corners <= 3 && max_vertices != 0 && max_primitives != 0 &&
               words_per_workgroup() <= std::numeric_limits<uint32_t>::max();
    }
};

} // namespace mkwvk
