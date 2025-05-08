#pragma once

#include <mlx/array.h>
#include <mlx/stream.h>

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"

namespace mx = mlx::core;

namespace pal::cpp {

// Declaration of the C++ paged_attention operation function
mx::array paged_attention(
    const mx::array& q,
    const mx::array& kv_cache,
    const mx::array& page_table,
    const mx::Stream& s
);

} // namespace pal::cpp
