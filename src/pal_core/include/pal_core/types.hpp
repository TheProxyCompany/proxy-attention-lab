#pragma once
// types.hpp
// Common types used across PAL kernels
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <cstdint>
#include <cstddef>

namespace pal::cpp {

// Core dimensions extracted from kernel inputs
struct CoreDims {
    uint32_t head_dim{0};
    uint32_t num_q_heads{0};
    uint32_t tokens_per_page{0};
    uint32_t num_kv_heads{0};
    size_t num_items_to_process{0};
    size_t query_token_count{0};
};

// Configuration for kernel dispatch
struct DispatchConfig {
    size_t grid_width{1};
    size_t grid_height{1};
    size_t grid_depth{1};
    size_t threads_per_group{64};
    size_t threadgroup_memory_bytes{0};
};

// Memory constraints for kernel execution
struct MemoryBudget {
    size_t total_available;
    size_t fixed_overhead;
    size_t dynamic_per_item;

    size_t remaining() const {
        return (total_available > fixed_overhead) ?
               (total_available - fixed_overhead) : 0;
    }
};

// Tile configuration for blocked algorithms
struct TileStrategy {
    uint32_t tile_size_m{0};
    uint32_t tile_size_n{0};
    uint32_t tile_size_k{0};

    bool is_valid() const {
        return tile_size_m > 0 && tile_size_n > 0 && tile_size_k > 0;
    }
};

// Performance hints for kernel optimization
struct PerformanceHints {
    bool prefer_small_tiles{false};
    bool maximize_occupancy{true};
    bool minimize_memory_traffic{false};
    uint32_t target_waves_per_eu{0};
};

} // namespace pal::cpp
