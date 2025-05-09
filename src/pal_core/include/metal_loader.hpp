#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <iostream>

// Forward declare for platform-specific path finding
std::string find_own_shared_library_path();

namespace pal::core::detail {

class MetalLibRegistrar {
public:
    static void ensure_pal_metallib_registered() {
        static std::atomic<bool> s_registered = false;
        static std::mutex s_mutex;

        // Fast path: already registered
        if (s_registered.load(std::memory_order_acquire)) {
            return;
        }

        std::lock_guard<std::mutex> lock(s_mutex);
        // Double-check after acquiring lock
        if (s_registered.load(std::memory_order_relaxed)) {
            return;
        }

        std::cerr << "[Debug PAL MetalLoader] Attempting to register pal.metallib..." << std::endl;

        std::string lib_path_str = find_own_shared_library_path();
        if (lib_path_str.empty()) {
            std::cerr << "ERROR [PAL MetalLoader]: Could not determine path of pal_core shared library." << std::endl;
            // Optionally throw, or let downstream kernel loading fail
            // For now, we'll let it proceed and fail at kernel load if path is bad.
            // A more robust solution would be to throw here if path is critical.
            // throw std::runtime_error("Could not determine path of pal_core shared library for Metal registration.");
            return; // Or throw
        }

        std::filesystem::path shared_lib_path(lib_path_str);
        std::filesystem::path metallib_dir = shared_lib_path.parent_path();
        std::filesystem::path full_metallib_path = metallib_dir / "pal.metallib";

        if (!std::filesystem::exists(full_metallib_path)) {
            std::cerr << "ERROR [PAL MetalLoader]: pal.metallib not found at expected location: "
                      << full_metallib_path.string() << std::endl;
            // throw std::runtime_error("pal.metallib not found at: " + full_metallib_path.string());
            return; // Or throw
        }

        std::string metallib_to_register = full_metallib_path.string();
        const std::string library_name = "pal";

        try {
            // Assuming mx::Device::gpu is the default target for paged attention
            auto& metal_device = mx::metal::device(mx::Device::gpu);
            std::cerr << "[Debug PAL MetalLoader] Registering Metal library '" << library_name
                      << "' from path: " << metallib_to_register << std::endl;
            metal_device.register_library(library_name, metallib_to_register);
            std::cerr << "[Debug PAL MetalLoader] Successfully registered Metal library '" << library_name << "'" << std::endl;
            s_registered.store(true, std::memory_order_release);
        } catch (const std::exception& e) {
            std::cerr << "ERROR [PAL MetalLoader] during pal.metallib registration: " << e.what() << std::endl;
            // throw std::runtime_error(std::string("Failed to register pal.metallib: ") + e.what());
            // Decide if this is fatal for the library's usability.
        }
    }
};

} // namespace pal::core::detail
