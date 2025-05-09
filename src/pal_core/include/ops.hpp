#pragma once

#include <mlx/array.h>
#include <mlx/utils.h>

namespace mx = mlx::core;

namespace pal::cpp {

// Declaration of the C++ paged_attention operation function
mx::array paged_attention(
    const mx::array& queries,
    const mx::array& kv_cache,
    const mx::array& page_table
    // mx::StreamOrDevice stream = {}
);

} // namespace pal::cpp
