#include "pal_core/ops.hpp"
#include "pal_core/paged_attention_primitive.hpp"
#include "pal_core/metal_loader.hpp"
#include <iostream>
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

namespace pal::cpp {

mx::array paged_attention(
    const mx::array& queries,
    const mx::array& k_cache_pool,
    const mx::array& v_cache_pool,
    const mx::array& page_table,
    const mx::array& sequence_lengths,
    const mx::array& query_to_seq_map,
    const mx::array& query_token_offset,
    mx::StreamOrDevice stream_or_device
) {
#ifdef PAL_DEBUG
    std::cerr << "[PAL Ops] pal::cpp::paged_attention C++ operation called." << std::endl;
#endif

    pal::core::detail::MetalLibRegistrar::ensure_pal_metallib_registered(stream_or_device);

    // Extract key parameters from input arrays to pass to the primitive
    int num_q_heads = 1;  // Default for 1D/2D queries
    int head_dim = 0;
    int tokens_per_page = 0;
    int num_kv_heads = 0;

    // Extract head_dim and tokens_per_page from K cache pool
    if (k_cache_pool.ndim() == 4) {
        tokens_per_page = k_cache_pool.shape(1);
        num_kv_heads = k_cache_pool.shape(2);
        head_dim = k_cache_pool.shape(3);
    }

    // For 3D queries, num_q_heads comes from the second dimension
    if (queries.ndim() == 3) {
        num_q_heads = queries.shape(1);
    }

#ifdef PAL_DEBUG
    std::cerr << "[PAL Ops] Creating primitive with extracted params: "
              << "num_q_heads=" << num_q_heads
              << ", num_kv_heads=" << num_kv_heads
              << ", head_dim=" << head_dim
              << ", tokens_per_page=" << tokens_per_page
              << std::endl;
#endif

    // Create the primitive instance with the extracted parameters
    auto primitive = std::make_shared<PagedAttentionPrimitive>(
        stream_or_device,
        num_q_heads,
        num_kv_heads,
        head_dim,
        tokens_per_page
    );

#ifdef PAL_DEBUG
    std::cerr << "[PAL Ops] PagedAttentionPrimitive instance created." << std::endl;
#endif

    // Use the primitive's output_shapes method to determine the correct output shape
    auto output_shapes = primitive->output_shapes({queries, k_cache_pool, v_cache_pool, page_table,
                                                 sequence_lengths, query_to_seq_map, query_token_offset});
    if (output_shapes.empty()) {
        throw std::runtime_error("[PAL Ops] PagedAttentionPrimitive returned empty output_shapes");
    }
    auto out_shape = output_shapes[0];
    auto out_dtype = queries.dtype();

#ifdef PAL_DEBUG
    std::cerr << "[PAL Ops] Output shape determined from primitive: [";
    for (size_t i = 0; i < out_shape.size(); ++i) {
        std::cerr << out_shape[i];
        if (i < out_shape.size() - 1) std::cerr << ", ";
    }
    std::cerr << "]" << std::endl;
#endif

    // Construct the output MLX array, adding the operation in the graph
    return mx::array(
        out_shape,
        out_dtype,
        primitive,
        {queries,
         k_cache_pool,
         v_cache_pool,
         page_table,
         sequence_lengths,
         query_to_seq_map,
         query_token_offset}
    );
}

} // namespace pal::cpp
