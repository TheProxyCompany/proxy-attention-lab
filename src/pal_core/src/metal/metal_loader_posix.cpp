// metal_loader_posix.cpp
// POSIX implementation of shared library path detection for PAL.
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

#include "pal_core/metal/metal_loader.hpp"

#include <dlfcn.h>
#include <string>
#include <spdlog/spdlog.h>

/**
 * @brief Find the path to the PAL shared library on POSIX systems
 *
 * Uses dladdr to identify the path of the shared library containing PAL code.
 * This is a POSIX-specific implementation that works on macOS and Linux.
 *
 * @return std::string Path to the shared library or empty string if not found
 */
std::string find_own_shared_library_path_for_pal() {
    Dl_info info{};
    // dladdr gives us info about the shared object containing this function
    if (dladdr(reinterpret_cast<void*>(&find_own_shared_library_path_for_pal), &info) == 0) {
        // dladdr failed: get error string if available
        const char* err = dlerror();
        spdlog::error("[PAL] dladdr() failed");
        if (err) spdlog::error("{}", err);
        return "";
    }

    if (!info.dli_fname) {
        spdlog::error("[PAL] dladdr() succeeded but dli_fname is null (no shared library path found)");
        return "";
    }

    // dli_fname is the path to the shared object containing this function
    return std::string(info.dli_fname);
}
