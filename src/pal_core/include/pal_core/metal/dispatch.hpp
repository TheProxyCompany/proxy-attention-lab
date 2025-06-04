#pragma once
// dispatch.hpp
// Metal dispatch utilities for kernel execution
//
// Copyright 2025 The Proxy Company. All Rights Reserved.

#include <mlx/array.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#include <Metal/Metal.hpp>
#include <vector>
#include <string>

namespace mx = mlx::core;

namespace pal::cpp::metal {

// Thread configuration for kernel dispatch
struct ThreadConfig {
    size_t threads_per_group;
    size_t execution_width;
    size_t max_threads_device;
};

// Dispatch grid configuration
struct DispatchGrid {
    size_t width;
    size_t height;
    size_t depth;

    DispatchGrid(size_t w = 1, size_t h = 1, size_t d = 1)
        : width(w), height(h), depth(d) {}

    MTL::Size to_mtl_size() const {
        return MTL::Size(width, height, depth);
    }
};

// Helper class for Metal kernel dispatch
class MetalDispatcher {
public:
    // Calculate optimal thread configuration for a kernel
    static ThreadConfig calculate_optimal_threads(
        MTL::ComputePipelineState* kernel_state,
        size_t target_threads = 64
    );

    // Helper to dispatch a kernel with common setup
    static void dispatch_kernel(
        mx::metal::CommandEncoder& encoder,
        const DispatchGrid& grid,
        size_t threads_per_group,
        size_t threadgroup_memory_bytes = 0
    );
};

} // namespace pal::cpp::metal
