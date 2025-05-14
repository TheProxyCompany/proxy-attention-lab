// metal_loader_posix.cpp
// POSIX implementation of shared library path detection for PAL.
//
// Copyright 2024 The Proxy Company. All Rights Reserved.
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

#include "pal_core/metal_loader.hpp"

#include <dlfcn.h>
#include <iostream>
#include <string>

/**
 * @brief Find the path to the PAL shared library on POSIX systems
 *
 * Uses dladdr to identify the path of the shared library containing PAL code.
 * This is a POSIX-specific implementation that works on macOS and Linux.
 *
 * @return std::string Path to the shared library or empty string if not found
 */
std::string find_own_shared_library_path_for_pal() {
  Dl_info dl_info;

  // Get information about this function's address in memory
  if (dladdr(reinterpret_cast<void*>(find_own_shared_library_path_for_pal),
             &dl_info)) {
    if (dl_info.dli_fname) {
      return std::string(dl_info.dli_fname);
    } else {
      std::cerr << "ERROR [PAL dladdr] dli_fname was null, cannot determine "
                   "shared library path."
                << std::endl;
    }
  } else {
    const char* dlsym_error = dlerror();
    std::cerr << "ERROR [PAL dladdr] dladdr() call failed";
    if (dlsym_error) {
      std::cerr << ". Error: " << dlsym_error;
    }
    std::cerr << std::endl;
  }

  return "";
}
