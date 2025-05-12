#pragma once

#include <mlx/array.h>
#include <mlx/utils.h>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace pal::cpp {

// Declaration of the C++ paged_attention operation function
mx::array paged_attention(
    const mx::array& queries,
    const mx::array& k_cache_pool,        // Global K data
    const mx::array& v_cache_pool,        // Global V data
    const mx::array& page_table,          // Contains physical_page_ids
    const mx::array& sequence_lengths,    // Actual length of each seq in batch
    const mx::array& query_to_seq_map,    // Maps global query token index to its seq_idx_in_batch
    const mx::array& query_token_offset,  // Logical offset of Q token within its sequence
    mx::StreamOrDevice stream
);

} // namespace pal::cpp
