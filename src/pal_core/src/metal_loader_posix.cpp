#include "pal_core/metal_loader.hpp"
#include <dlfcn.h>
#include <string>
#include <iostream>

std::string find_own_shared_library_path_for_pal() {
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void*>(find_own_shared_library_path_for_pal), &dl_info)) {
        if (dl_info.dli_fname) {
            return std::string(dl_info.dli_fname);
        } else {
            std::cerr << "ERROR [PAL dladdr] dli_fname was null, cannot determine shared library path." << std::endl;
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
