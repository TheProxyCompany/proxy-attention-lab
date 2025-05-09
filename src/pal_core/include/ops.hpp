#pragma once

#include <mlx/array.h>
#include <mlx/utils.h>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace pal::cpp {

// Declaration of the C++ paged_attention operation function
int paged_attention(
    mx::array queries
    // const mx::array& kv_cache,
    // const mx::array& page_table
    // mx::StreamOrDevice stream = {}
);

} // namespace pal::cpp
