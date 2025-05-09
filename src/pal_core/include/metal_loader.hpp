#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <iostream>
#include <filesystem>

#include <mlx/mlx.h>
#include <mlx/device.h>
#include "mlx/backend/metal/device.h"

namespace mx = mlx::core;
// Forward declare for platform-specific path finding
std::string find_own_shared_library_path();

namespace pal::core::detail {

class MetalLibRegistrar {
public:
    static void ensure_pal_metallib_registered() {
        static std::atomic<bool> s_registered = false;
        static std::mutex s_mutex;

        if (s_registered.load(std::memory_order_acquire)) {
            return;
        }

        std::lock_guard<std::mutex> lock(s_mutex);
        if (s_registered.load(std::memory_order_relaxed)) {
            return;
        }

        std::cerr << "[Debug PAL MetalLoader] Attempting to register pal.metallib..." << std::endl;

        std::string lib_path_str = find_own_shared_library_path();
        if (lib_path_str.empty()) {
            std::cerr << "ERROR [PAL MetalLoader]: Could not determine path of pal_core shared library." << std::endl;
            throw std::runtime_error("Could not determine path of pal_core shared library for Metal registration.");
        }

        std::filesystem::path shared_lib_path(lib_path_str);
        std::filesystem::path metallib_dir = shared_lib_path.parent_path();
        std::filesystem::path full_metallib_path = metallib_dir / "pal.metallib";

        if (!std::filesystem::exists(full_metallib_path)) {
            std::cerr << "ERROR [PAL MetalLoader]: pal.metallib not found at expected location: "
                      << full_metallib_path.string() << std::endl;
            throw std::runtime_error("pal.metallib not found at: " + full_metallib_path.string());
        }

        std::string metallib_to_register = full_metallib_path.string();
        const std::string library_name = "pal";

        try {
            auto& metal_device = mx::metal::device(mx::Device::gpu);
            std::cerr << "[Debug PAL MetalLoader] Registering Metal library '" << library_name
                      << "' from path: " << metallib_to_register << std::endl;
            metal_device.register_library(library_name, metallib_to_register);
            std::cerr << "[Debug PAL MetalLoader] Successfully registered Metal library '" << library_name << "'" << std::endl;
            s_registered.store(true, std::memory_order_release);
        } catch (const std::exception& e) {
            std::cerr << "ERROR [PAL MetalLoader] during pal.metallib registration: " << e.what() << std::endl;
            throw std::runtime_error(std::string("Failed to register pal.metallib: ") + e.what());
        }
    }
};

} // namespace pal::core::detail
