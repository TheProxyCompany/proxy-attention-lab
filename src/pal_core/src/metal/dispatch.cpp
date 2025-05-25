// dispatch.cpp
// Implementation of Metal dispatch utilities
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

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

void MetalDispatcher::setup_input_arrays(
    mx::metal::CommandEncoder& encoder,
    const std::vector<mx::array>& inputs,
    size_t starting_index
) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        encoder.set_input_array(inputs[i], starting_index + i);
    }
}

void MetalDispatcher::validate_input_pointers(
    const std::vector<mx::array>& inputs,
    const std::string& kernel_name
) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i].data<void>()) {
            spdlog::error("[{}] Input array {} has null data pointer", kernel_name, i);
            throw std::runtime_error("Null input data pointer detected in " + kernel_name);
        }
    }
}

void MetalDispatcher::dispatch_kernel(
    mx::metal::CommandEncoder& encoder,
    const DispatchGrid& grid,
    const ThreadConfig& threads,
    size_t threadgroup_memory_bytes
) {
    MTL::Size threadgroups_per_grid = grid.to_mtl_size();
    MTL::Size threads_per_threadgroup = MTL::Size(threads.threads_per_group, 1, 1);

    if (threadgroup_memory_bytes > 0) {
        encoder.set_threadgroup_memory_length(threadgroup_memory_bytes, 0);
    }

    encoder.dispatch_threadgroups(threadgroups_per_grid, threads_per_threadgroup);
}

} // namespace pal::cpp::metal
