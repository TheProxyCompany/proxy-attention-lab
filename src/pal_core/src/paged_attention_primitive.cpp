#include "paged_attention_primitive.hpp"
#include <mlx/allocator.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/backend/cpu/encoder.h>

#include <stdexcept>
#include <iostream>

namespace pal::cpp {

// --- Constructor ---
PagedAttentionPrimitive::PagedAttentionPrimitive(mx::StreamOrDevice stream_or_device)
    : mx::UnaryPrimitive(mx::to_stream(stream_or_device)) { // Correctly use the passed stream/device
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive constructed." << std::endl;
}

// --- CPU Evaluation (Stub) ---
void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive::eval_cpu called (not supported)." << std::endl;
    throw std::runtime_error("[PagedAttentionPrimitive] CPU evaluation is not supported for paged attention.");
}

// --- GPU Evaluation (Kernel Launch Logic) ---
void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::cerr << "[PAL Primitive] PagedAttentionPrimitive::eval_gpu called." << std::endl;

    // Input validation (as before)
    if (inputs.size() != 3) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::eval_gpu] Expected 3 inputs (Q, KV_cache, Page_table), received " +
            std::to_string(inputs.size()));
    }
    const auto& q = inputs[0];
    const auto& kv_cache = inputs[1];
    const auto& page_table = inputs[2];

    if (q.dtype() != mx::float16 || kv_cache.dtype() != mx::float16) {
         throw std::invalid_argument("[PagedAttentionPrimitive::eval_gpu] Kernel requires float16 Q and KV cache.");
    }
    if (page_table.dtype() != mx::uint32) {
         throw std::invalid_argument("[PagedAttentionPrimitive::eval_gpu] Page table must be uint32.");
    }

    // Allocate output buffer (MLX ensures `out` has correct shape/dtype from output_shapes())
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::cerr << "[PAL Primitive] Output buffer allocated." << std::endl;

    // Metal Kernel Invocation Logic
    auto& s = stream();
    auto& d = mlx::core::metal::device(mx::Device::gpu);

    const std::string library_name_for_mlx = "pal"; // Must match name used in MetalLibRegistrar
    const std::string kernel_name = "paged_attn_kernel"; // Matches [[kernel]] name in .metal file

    std::cerr << "[PAL Primitive] Attempting to get kernel '" << kernel_name
              << "' from MLX library '" << library_name_for_mlx
              << "' (must be pre-registered)." << std::endl;

    MTL::ComputePipelineState* kernel = nullptr;
    try {
        kernel = d.get_kernel(kernel_name, library_name_for_mlx);
    } catch (const std::runtime_error& e) {
        std::cerr << "[PAL Primitive] Error: " << e.what() << std::endl;
        throw std::runtime_error("PagedAttentionPrimitive: Failed to get kernel '" + kernel_name +
                                 "' from library '" + library_name_for_mlx +
                                 "'. Ensure the 'pal.metallib' was registered successfully.");
    }
    if (!kernel) {
         throw std::runtime_error("PagedAttentionPrimitive: Failed to get kernel '" + kernel_name +
                                  "' from library '" + library_name_for_mlx +
                                  "'. Ensure the 'pal.metallib' was registered successfully.");
    }
    std::cerr << "[PAL Primitive] Metal kernel object retrieved." << std::endl;

    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    std::cerr << "[PAL Primitive] Compute pipeline state set." << std::endl;

    // Set kernel arguments (as before)
    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(kv_cache, 1);
    compute_encoder.set_input_array(page_table, 2);
    compute_encoder.set_output_array(out, 3);
    std::cerr << "[PAL Primitive] Kernel buffers set." << std::endl;

    // Calculate grid/threadgroup dimensions (as before)
    size_t grid_dim_x = q.size();
    if (grid_dim_x == 0) {
        std::cerr << "[PAL Primitive Warning] Input 'q' is empty, skipping kernel dispatch." << std::endl;
        return;
    }
    MTL::Size grid_dims = MTL::Size(grid_dim_x, 1, 1);
    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    if (tgp_size == 0) { // Should not happen for a valid kernel
        std::cerr << "[PAL Primitive Warning] Kernel maxTotalThreadsPerThreadgroup is 0. Defaulting to 1." << std::endl;
        tgp_size = 1;
    }
    tgp_size = std::min(tgp_size, grid_dim_x); // Cap at grid_dim_x if grid is smaller
    if (grid_dim_x > 0 && tgp_size == 0) tgp_size = 1; // Ensure tgp_size is at least 1 if dispatching

    MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
    std::cerr << "[PAL Primitive] Dispatching kernel: grid_dim_x=" << grid_dim_x
              << ", threadgroup_size=" << tgp_size << std::endl;

    compute_encoder.dispatch_threads(grid_dims, group_dims);
    std::cerr << "[PAL Primitive] Kernel dispatched." << std::endl;
}

// --- Print Method ---
void PagedAttentionPrimitive::print(std::ostream& os) {
    os << "PagedAttention";
    // Add any parameters if the primitive stores them, e.g.:
    // os << "(block_size=" << block_size_ << ")";
}

// --- Equivalence Check ---
bool PagedAttentionPrimitive::is_equivalent(const mx::Primitive& other) const {
    // Check if the other primitive is the same type
    if (typeid(*this) != typeid(other)) {
        return false;
    }
    // Cast and compare any parameters if they exist
    // const PagedAttentionPrimitive& other_pa = static_cast<const PagedAttentionPrimitive&>(other);
    // return this->block_size_ == other_pa.block_size_; // Example
    return true; // No parameters stored currently
}

// --- Gradient Stubs ---
std::vector<mx::array> PagedAttentionPrimitive::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
    // VJP for attention is complex. Stub for now.
    throw std::runtime_error("[PagedAttentionPrimitive] VJP not implemented.");
}

std::vector<mx::array> PagedAttentionPrimitive::jvp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& tangents,
    const std::vector<int>& argnums) {
    // JVP for attention is complex. Stub for now.
    throw std::runtime_error("[PagedAttentionPrimitive] JVP not implemented.");
}

// --- Vmap Stub ---
std::pair<std::vector<mx::array>, std::vector<int>> PagedAttentionPrimitive::vmap(
    const std::vector<mx::array>& inputs,
    const std::vector<int>& axes) {
    // Vmap for attention might require careful handling of batch/head dimensions.
    throw std::runtime_error("[PagedAttentionPrimitive] Vmap not implemented.");
}

// --- Output Shape Calculation ---
std::vector<mx::Shape> PagedAttentionPrimitive::output_shapes(const std::vector<mx::array>& inputs) {
    // Assuming the output shape is the same as the query shape (Q)
    // This might need adjustment based on the final kernel logic (e.g., if it concatenates heads)
    if (inputs.empty()) {
         throw std::invalid_argument("[PagedAttentionPrimitive::output_shapes] Requires at least one input (query).");
    }
    // For the stub, output shape matches query shape
    return {inputs[0].shape()};
}

} // namespace pal::cpp
