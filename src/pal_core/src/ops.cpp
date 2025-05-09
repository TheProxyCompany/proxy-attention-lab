#include "ops.hpp"
#include "paged_attention_primitive.hpp"
#include <iostream>
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"

namespace pal::cpp {

// Define the user-facing C++ operation function
mx::array paged_attention(
    const mx::array& queries,
    const mx::array& kv_cache,
    const mx::array& page_table,
    mx::StreamOrDevice stream
) {
    std::cerr << "[Debug] pal::cpp::paged_attention C++ operation called" << std::endl;

    // --- Determine Output Shape ---
    // For the stub, assume output shape matches query shape.
    // A real implementation might calculate based on head concatenation, etc.
    auto out_shape = queries.shape();
    auto out_dtype = queries.dtype(); // Output dtype usually matches query/value dtype

    // --- Create Primitive Instance ---
    // Pass the stream/device context to the primitive
    auto primitive = std::make_shared<PagedAttentionPrimitive>(mx::to_stream({}));
    std::cerr << "[Debug] paged_attention: Created PagedAttentionPrimitive instance" << std::endl;


    // --- Construct Output Array ---
    // Create the output array object. This adds the operation to the graph.
    // The actual computation happens later when mx::eval() is called.
    return mx::array(
        out_shape,                        // Shape of the output
        out_dtype,                        // Dtype of the output
        primitive,                        // The primitive performing the computation
        {queries, kv_cache, page_table}         // Inputs to the primitive
    );
}

} // namespace pal::cpp
