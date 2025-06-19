#pragma once

#include <cstddef>

namespace pal::cpp {
// Page size used for paged attention operations
constexpr int PAGE_SIZE = 16;

inline size_t get_optimal_page_size() {
    return PAGE_SIZE;
}

} // namespace pal::cpp
