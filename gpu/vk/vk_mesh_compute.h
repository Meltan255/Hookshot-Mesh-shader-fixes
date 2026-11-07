// RX 580 mesh-shader fallback output layout.
#pragma once

#include <cstdint>

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

} // namespace mkwvk
