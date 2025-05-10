#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <iostream>
#include <filesystem>

#include <mlx/device.h>
#include <mlx/stream.h>
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

#include "paged_attention_primitive.hpp"
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
// #include <mlx/backend/>

#include <stdexcept>
#include <iostream>

namespace mx = mlx::core;

// Forward declaration
std::string find_own_shared_library_path_for_pal();

namespace pal::core::detail {

class MetalLibRegistrar {
public:
    static void ensure_pal_metallib_registered(const mx::StreamOrDevice& stream_or_device) {
        static std::atomic<bool> s_registered = false;
        static std::mutex s_mutex;

        if (s_registered.load(std::memory_order_acquire)) {
            return;
        }

        std::lock_guard<std::mutex> lock(s_mutex);
        if (s_registered.load(std::memory_order_relaxed)) {
            return;
        }

        std::cerr << "[PAL MetalLoader] Attempting one-time registration..." << std::endl;

        std::string own_lib_path_str = find_own_shared_library_path_for_pal();
        if (own_lib_path_str.empty()) {
            std::cerr << "ERROR [PAL MetalLoader]: Could not determine path of the pal_core shared library. PAL Metal kernels will be unavailable." << std::endl;
            return;
        }
        std::cerr << "[PAL MetalLoader] pal_core shared library path: \"" << own_lib_path_str << "\"" << std::endl;

        std::filesystem::path shared_lib_path(own_lib_path_str);
        std::filesystem::path metallib_dir = shared_lib_path.parent_path();
        const std::string desired_mlx_alias = "pal";

        try {
            auto& d = mx::metal::device(mx::to_stream(stream_or_device).device);

            std::cerr << "[PAL MetalLoader] Calling metal_device.register_library with:" << std::endl;
            std::cerr << "  MLX Alias (arg1 lib_name): \"" << desired_mlx_alias << "\"" << std::endl;

            d.register_library(desired_mlx_alias, metallib_dir.string());

            std::cerr << "[PAL MetalLoader] Call to register_library completed without throwing an exception." << std::endl;
            std::cerr << "  Registered with MLX Alias: \"" << desired_mlx_alias << "\"" << std::endl;
            std::cerr << "  Using File Path: \"" << metallib_dir << "\"" << std::endl;

            s_registered.store(true, std::memory_order_release);

        } catch (const std::exception& e) {
            std::cerr << "ERROR [PAL MetalLoader] Exception during metal_device.register_library call: " << e.what() << std::endl;
            std::cerr << "  Attempted MLX Alias: \"" << desired_mlx_alias << "\"" << std::endl;
            std::cerr << "  Attempted File Path: \"" << metallib_dir << "\"" << std::endl;
        }
    }
};

} // namespace pal::core::detail
