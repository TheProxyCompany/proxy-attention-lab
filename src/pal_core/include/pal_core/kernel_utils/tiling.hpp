#pragma once
// tiling.hpp
// Utilities for calculating optimal tile sizes for kernels
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace pal::cpp::tiling {

// Requirements for tile size calculation
struct TileRequirements {
    size_t bytes_per_element;
    uint32_t alignment;
    uint32_t min_size;
    uint32_t max_size;
    uint32_t preferred_size;
};

// Configuration for calculated tile size
struct TileConfig {
    uint32_t tile_size;
    size_t memory_usage;

    bool is_valid() const { return tile_size > 0; }
};

// Calculate optimal tile size given memory constraints
inline TileConfig calculate_optimal_tile_size(
    size_t available_memory,
    const TileRequirements& requirements
) {
    TileConfig config{0, 0};

    if (available_memory == 0 || requirements.bytes_per_element == 0) {
        return config;
    }

    // Calculate maximum possible tile size that fits in memory
    uint32_t max_fit = static_cast<uint32_t>(available_memory / requirements.bytes_per_element);
    max_fit = std::min(max_fit, requirements.max_size);

    // Align down to required alignment
    if (requirements.alignment > 0) {
        max_fit = (max_fit / requirements.alignment) * requirements.alignment;
    }

    // If we can fit the preferred size, use it
    if (max_fit >= requirements.preferred_size) {
        config.tile_size = requirements.preferred_size;
    } else if (max_fit >= requirements.min_size) {
        // Otherwise use the maximum that fits
        config.tile_size = max_fit;
    } else if (max_fit > 0 && requirements.alignment > 0) {
        // If we can't meet minimum, try to at least use alignment size
        config.tile_size = requirements.alignment;
    }

    // Calculate actual memory usage
    config.memory_usage = config.tile_size * requirements.bytes_per_element;

    return config;
}

// Helper for calculating tile sizes for 2D data (e.g., K and V tiles in attention)
struct TileConfig2D {
    uint32_t tile_size_rows;
    uint32_t tile_size_cols;
    size_t total_memory_usage;

    bool is_valid() const { return tile_size_rows > 0 && tile_size_cols > 0; }
};

} // namespace pal::cpp::tiling
