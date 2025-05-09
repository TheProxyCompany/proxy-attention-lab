#include "ops.hpp"
#include "paged_attention_primitive.hpp"
#include "metal_loader.hpp"
#include <iostream>
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

namespace pal::cpp {

mx::array paged_attention(
    const mx::array& queries,
    const mx::array& kv_cache,
    const mx::array& page_table,
    mx::StreamOrDevice stream_or_device
) {
    std::cerr << "[PAL Ops] pal::cpp::paged_attention C++ operation called." << std::endl;

    pal::core::detail::MetalLibRegistrar::ensure_pal_metallib_registered(stream_or_device);

    // Determine output shape and dtype based on inputs (example: matches queries)
    auto out_shape = queries.shape();
    auto out_dtype = queries.dtype();

    // Create the primitive instance, passing the specific stream/device for this operation
    auto primitive = std::make_shared<PagedAttentionPrimitive>(stream_or_device);
    std::cerr << "[PAL Ops] PagedAttentionPrimitive instance created." << std::endl;

    // Construct the output MLX array, adding the operation in the graph
    return mx::array(
        out_shape,
        out_dtype,
        primitive,
        {queries, kv_cache, page_table}
    );
}

} // namespace pal::cpp
