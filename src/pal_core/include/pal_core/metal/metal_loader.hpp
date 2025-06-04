#pragma once
// metal_loader.hpp
// Metal library loading and registration utilities for PAL.
//
// Copyright 2025 The Proxy Company. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include <atomic>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>

#include <mlx/mlx.h>
#include <mlx/device.h>
#include <mlx/stream.h>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace mx = mlx::core;

/**
 * @brief Locates the path to the PAL shared library
 *
 * Platform-specific implementation to find the path to the loaded shared library
 * containing PAL code, which is used to locate the associated Metal library.
 *
 * @return std::string Path to the shared library or empty string if not found
 */
std::string find_own_shared_library_path_for_pal();

namespace pal::cpp {

/**
 * @brief Handles Metal library registration for PAL kernels
 *
 * This class ensures that the Metal library containing PAL kernels is properly
 * registered with MLX's Metal backend. It performs a thread-safe, one-time
 * registration process.
 */
class MetalLibRegistrar {
 public:
  /**
   * @brief Ensures PAL Metal library is registered with MLX
   *
   * This method checks if the Metal library is already registered, and if not,
   * performs the registration with appropriate thread safety. It locates the
   * Metal library based on the location of the shared library.
   *
   * @param stream_or_device MLX stream or device to register the library with
   */
  static void ensure_pal_metallib_registered(
      const mx::StreamOrDevice& stream_or_device) {
    static std::atomic<bool> s_registered = false;
    static std::mutex s_mutex;

    // Fast path - already registered
    if (s_registered.load(std::memory_order_acquire)) {
      return;
    }

    // Slow path with locking
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_registered.load(std::memory_order_relaxed)) {
      return;
    }

    spdlog::debug("[PAL MetalLoader] Attempting one-time registration...");

    // Find the path to our own shared library
    std::string own_lib_path_str = find_own_shared_library_path_for_pal();
    if (own_lib_path_str.empty()) {
      spdlog::error("Could not determine path of the pal_core shared library. PAL Metal kernels will be unavailable.");
      return;
    }
    spdlog::debug("[PAL MetalLoader] pal_core shared library path: \"{}\"", own_lib_path_str);

    // Set up paths and names
    std::filesystem::path shared_lib_path(own_lib_path_str);
    std::filesystem::path metallib_dir = shared_lib_path.parent_path();
    const std::string kDesiredMlxAlias = "pal";

    try {
      // Get MLX Metal device
      auto& d = mx::metal::device(mx::to_stream(stream_or_device).device);

      spdlog::debug("[PAL MetalLoader] Calling metal_device.register_library with:");
      spdlog::debug("  MLX Alias (arg1 lib_name): \"{}\"", kDesiredMlxAlias);

      // Register the library
      d.register_library(kDesiredMlxAlias, metallib_dir.string());

      spdlog::debug("[PAL MetalLoader] Call to register_library completed without throwing an exception.");
      spdlog::debug("  Registered with MLX Alias: \"{}\"", kDesiredMlxAlias);
      spdlog::debug("  Using File Path: \"{}\"", metallib_dir.string());

      // Mark as registered
      s_registered.store(true, std::memory_order_release);

    } catch (const std::exception& e) {
      spdlog::error("Exception during metal_device.register_library call: {}", e.what());
      spdlog::error("  Attempted MLX Alias: \"{}\"", kDesiredMlxAlias);
      spdlog::error("  Attempted File Path: \"{}\"", metallib_dir.string());
    }
  }
};

}  // namespace pal::core::detail
