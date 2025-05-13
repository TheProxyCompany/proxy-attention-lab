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
    std::cerr << "[PAL Ops] pal::cpp::paged_attention C++ operation called." << std::endl;

    pal::core::detail::MetalLibRegistrar::ensure_pal_metallib_registered(stream_or_device);

    // Create the primitive instance, passing the specific stream/device for this operation
    auto primitive = std::make_shared<PagedAttentionPrimitive>(stream_or_device);
    std::cerr << "[PAL Ops] PagedAttentionPrimitive instance created." << std::endl;

    // Use the primitive's output_shapes method to determine the correct output shape
    auto output_shapes = primitive->output_shapes({queries, k_cache_pool, v_cache_pool, page_table,
                                                 sequence_lengths, query_to_seq_map, query_token_offset});
    if (output_shapes.empty()) {
        throw std::runtime_error("[PAL Ops] PagedAttentionPrimitive returned empty output_shapes");
    }
    auto out_shape = output_shapes[0];
    auto out_dtype = queries.dtype();

    std::cerr << "[PAL Ops] Output shape determined from primitive: [";
    for (size_t i = 0; i < out_shape.size(); ++i) {
        std::cerr << out_shape[i];
        if (i < out_shape.size() - 1) std::cerr << ", ";
    }
    std::cerr << "]" << std::endl;

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
