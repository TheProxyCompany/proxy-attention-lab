#include "ops.hpp"
#include "paged_attention_primitive.hpp"
#include <iostream>

namespace pal::cpp {

// Define the user-facing C++ operation function
mx::array paged_attention(
    const mx::array& queries,
    const mx::array& kv_cache,
    const mx::array& page_table
    // mx::StreamOrDevice stream
) {
    std::cerr << "[Debug] pal::cpp::paged_attention C++ operation called" << std::endl;

    // --- Input Validation (Example) ---
    // Add more checks as needed based on kernel requirements
    if (queries.ndim() < 2 || kv_cache.ndim() < 1 || page_table.ndim() < 1) {
         throw std::invalid_argument("[paged_attention] Invalid input dimensions.");
    }
     if (queries.dtype() != mx::float16 || kv_cache.dtype() != mx::float16) {
         throw std::invalid_argument("[paged_attention] Stub requires float16 queries and KV cache.");
     }
     if (page_table.dtype() != mx::uint32) {
          throw std::invalid_argument("[paged_attention] Page table must be uint32.");
     }

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
