// kernel_debug.cpp
// Implementation of kernel debug utilities
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include "pal_core/debug/kernel_debug.hpp"
#include <spdlog/spdlog.h>
#include <sstream>
#include <iomanip>

namespace pal::cpp::debug {

void KernelDebugger::log_dispatch(
    const std::string& kernel_name,
    const metal::DispatchGrid& grid,
    const metal::ThreadConfig& threads
) {
    spdlog::debug("[{} Dispatch] Grid: {}x{}x{}, Threads: {} (width: {}, max: {})",
                  kernel_name,
                  grid.width, grid.height, grid.depth,
                  threads.threads_per_group,
                  threads.execution_width,
                  threads.max_threads_device);
}

void KernelDebugger::log_memory_usage(
    const std::string& kernel_name,
    size_t total_bytes,
    size_t available_bytes
) {
    spdlog::debug("[{} Memory] Using {} of {} available",
                  kernel_name,
                  format_memory_size(total_bytes),
                  format_memory_size(available_bytes));
}

void KernelDebugger::log_memory_layout(
    const std::string& kernel_name,
    const kernel_utils::AttentionMemoryLayout& layout
) {
    spdlog::debug("[{} Memory Layout] Total: {}",
                  kernel_name, format_memory_size(layout.total_bytes));
    spdlog::debug("  - Q shared memory: {}", format_memory_size(layout.q_shmem_bytes));
    spdlog::debug("  - Reduction scratch: {}", format_memory_size(layout.partial_reduce_scratch_bytes));
    spdlog::debug("  - K tile: {}", format_memory_size(layout.k_tile_bytes));
    spdlog::debug("  - V tile: {}", format_memory_size(layout.v_tile_bytes));
    spdlog::debug("  - Page table: {}", format_memory_size(layout.page_table_slice_bytes));
}

void KernelDebugger::validate_and_log_inputs(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const std::vector<std::string>& input_names
) {
    if (inputs.size() != input_names.size()) {
        spdlog::warn("[{} Debug] Input count mismatch: {} arrays, {} names",
                     kernel_name, inputs.size(), input_names.size());
        return;
    }

    spdlog::debug("[{} Inputs] {} arrays:", kernel_name, inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& arr = inputs[i];
        spdlog::debug("  - {}: shape={}, nbytes={}",
                      input_names[i],
                      format_shape(arr),
                      format_memory_size(arr.nbytes()));
    }
}

void KernelDebugger::log_tile_config(
    const std::string& kernel_name,
    uint32_t tile_size,
    const std::string& tile_name
) {
    spdlog::debug("[{} Tiling] {} tile size: {}", kernel_name, tile_name, tile_size);
}

void KernelDebugger::log_kernel_start(const std::string& kernel_name) {
    spdlog::debug("[{} Timing] Kernel execution started", kernel_name);
}

void KernelDebugger::log_kernel_end(const std::string& kernel_name) {
    spdlog::debug("[{} Timing] Kernel execution completed", kernel_name);
}

std::string KernelDebugger::format_shape(const mx::array& arr) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < arr.ndim(); ++i) {
        if (i > 0) oss << ", ";
        oss << arr.shape(i);
    }
    oss << "]";
    return oss.str();
}

std::string KernelDebugger::format_memory_size(size_t bytes) {
    std::ostringstream oss;
    if (bytes >= 1024 * 1024) {
        oss << std::fixed << std::setprecision(2)
            << (bytes / (1024.0 * 1024.0)) << " MB";
    } else if (bytes >= 1024) {
        oss << std::fixed << std::setprecision(2)
            << (bytes / 1024.0) << " KB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

} // namespace pal::cpp::debug
