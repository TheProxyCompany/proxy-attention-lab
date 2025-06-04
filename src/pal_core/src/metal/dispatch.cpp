// dispatch.cpp
// Implementation of Metal dispatch utilities
//
// Copyright 2025 The Proxy Company. All Rights Reserved.

#include "pal_core/metal/dispatch.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <stdexcept>

namespace pal::cpp::metal {

ThreadConfig MetalDispatcher::calculate_optimal_threads(
    MTL::ComputePipelineState* kernel_state,
    size_t target_threads
) {
    ThreadConfig config;
    config.execution_width = kernel_state->threadExecutionWidth();
    config.max_threads_device = kernel_state->maxTotalThreadsPerThreadgroup();

    // Align target threads to execution width
    config.threads_per_group = ((target_threads + config.execution_width - 1) /
                               config.execution_width) * config.execution_width;

    // Cap by device maximum
    config.threads_per_group = std::min(config.threads_per_group, config.max_threads_device);

    spdlog::debug("[Metal Dispatch] Thread config: threads_per_group={}, execution_width={}, max_threads={}",
                  config.threads_per_group, config.execution_width, config.max_threads_device);

    return config;
}

void MetalDispatcher::dispatch_kernel(
    mx::metal::CommandEncoder& encoder,
    const DispatchGrid& grid,
    size_t threads_per_group,
    size_t threadgroup_memory_bytes
) {
    MTL::Size threadgroups_per_grid = grid.to_mtl_size();
    MTL::Size threads_per_threadgroup = MTL::Size(threads_per_group, 1, 1);

    if (threadgroup_memory_bytes > 0) {
        encoder.set_threadgroup_memory_length(threadgroup_memory_bytes, 0);
    }

    encoder.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
}

} // namespace pal::cpp::metal
