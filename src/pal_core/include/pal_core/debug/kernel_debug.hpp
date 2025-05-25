#pragma once
// kernel_debug.hpp
// Debug utilities for kernel development and troubleshooting
//
// Copyright 2024 The Proxy Company. All Rights Reserved.

#include <string>
#include <vector>
#include <mlx/array.h>
#include "pal_core/metal/dispatch.hpp"
#include "pal_core/kernel_utils/memory_layout.hpp"

namespace mx = mlx::core;

namespace pal::cpp::debug {

// Centralized debug logging for kernel operations
class KernelDebugger {
public:
    // Log kernel dispatch configuration
    static void log_dispatch(
        const std::string& kernel_name,
        const metal::DispatchGrid& grid,
        const metal::ThreadConfig& threads
    );

    // Log memory usage for a kernel
    static void log_memory_usage(
        const std::string& kernel_name,
        size_t total_bytes,
        size_t available_bytes
    );

    // Log detailed memory layout breakdown
    static void log_memory_layout(
        const std::string& kernel_name,
        const kernel_utils::AttentionMemoryLayout& layout
    );

    // Validate and log input array information
    static void validate_and_log_inputs(
        const std::string& kernel_name,
        const std::vector<mx::array>& inputs,
        const std::vector<std::string>& input_names
    );

    // Log tile configuration
    static void log_tile_config(
        const std::string& kernel_name,
        uint32_t tile_size,
        const std::string& tile_name
    );

    // Performance timing helpers
    static void log_kernel_start(const std::string& kernel_name);
    static void log_kernel_end(const std::string& kernel_name);

private:
    // Format array shape for logging
    static std::string format_shape(const mx::array& arr);

    // Format memory size in human-readable format
    static std::string format_memory_size(size_t bytes);
};

} // namespace pal::cpp::debug
