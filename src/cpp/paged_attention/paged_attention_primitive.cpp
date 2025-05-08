#include "paged_attention_primitive.hpp"
#include <mlx/backend/metal/metal.h>
#include "mlx/backend/metal/utils.h"
#include <mlx/allocator.h>
#include <mlx/device.h>
#include <stdexcept>
#include <filesystem>
#include <iostream>

namespace pal::cpp {

// --- Constructor ---
PagedAttentionPrimitive::PagedAttentionPrimitive(mx::Stream stream)
    : mx::UnaryPrimitive(stream) {
    // Initialize any member variables here if needed
    std::cerr << "[Debug] PagedAttentionPrimitive constructed" << std::endl;
}

// --- CPU Evaluation (Stub) ---
void PagedAttentionPrimitive::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
    // Paged attention is GPU-specific
    std::cerr << "[Debug] PagedAttentionPrimitive::eval_cpu called" << std::endl;
    std::cerr << "[Debug] inputs: " << inputs.size() << std::endl;
    std::cerr << "[Debug] out size: " << out.shape().size() << std::endl;
    throw std::runtime_error("[PagedAttentionPrimitive] CPU evaluation is not supported.");
}

// --- GPU Evaluation (Kernel Launch Logic) ---
void PagedAttentionPrimitive::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
    std::cerr << "[Debug] PagedAttentionPrimitive::eval_gpu called" << std::endl;

    // Basic input validation
    // Expecting: Q, KV_cache_buffer, Page_table
    if (inputs.size() != 3) {
        throw std::invalid_argument(
            "[PagedAttentionPrimitive::eval_gpu] Expected 3 inputs (Q, KV_cache, Page_table), received " +
            std::to_string(inputs.size()));
    }
    const auto& q = inputs[0];
    const auto& kv_cache = inputs[1];
    const auto& page_table = inputs[2];

    // Validate dtypes (adjust as needed for kernel)
    if (q.dtype() != mx::float16 || kv_cache.dtype() != mx::float16) {
         throw std::invalid_argument("[PagedAttentionPrimitive::eval_gpu] Stub kernel requires float16 Q and KV cache.");
    }
     if (page_table.dtype() != mx::uint32) {
         throw std::invalid_argument("[PagedAttentionPrimitive::eval_gpu] Page table must be uint32.");
     }

    // --- Allocate Output Buffer ---
    // The output shape is determined by output_shapes() before eval_gpu is called.
    // MLX ensures `out` has the correct shape and dtype. We just need to allocate memory.
    // Using set_data ensures MLX manages the allocation lifecycle.
    out.set_data(mx::allocator::malloc(out.nbytes()));
    std::cerr << "[Debug] Allocated output buffer for PagedAttentionPrimitive" << std::endl;


    // --- Metal Kernel Invocation Logic (from previous binding.cpp) ---
    auto& s = stream(); // Get the stream associated with this primitive
    auto& d = mx::metal::device(s.device);

    // Find and register the Metal library
    // First try to find it relative to the current module
    std::string metallib_path;
    try {
        // Try to find it in the installed package
        // This relies on nanobind's nb::module_ context which isn't available here
        // So we use a simple heuristic approach
        auto module_path = std::filesystem::path(__FILE__).parent_path();
        auto potential_paths = {
            // In development mode, it's in the build dir
            std::filesystem::path(module_path) / ".." / "build" / "pal.metallib",
            // In installed mode, it's alongside the .so/.dylib
            std::filesystem::path(module_path) / "pal.metallib",
        };

        for (const auto& path : potential_paths) {
            if (std::filesystem::exists(path)) {
                metallib_path = path.string();
                break;
            }
        }

        if (metallib_path.empty()) {
            // Fallback to a hardcoded path as last resort
            metallib_path = "src/python/pie_paged_attn_lab.metallib";
        }
    } catch (const std::exception& e) {
        std::cerr << "[Warning] Error finding metallib: " << e.what() << std::endl;
        metallib_path = "src/python/pie_paged_attn_lab.metallib"; // Fallback
    }
    std::string library_name = "pal"; // Must match TITLE in mlx_build_metallib

    try {
         std::cerr << "[Debug] Registering library '" << library_name << "' from path: " << metallib_path << std::endl;
         d.register_library(library_name, metallib_path);
         std::cerr << "[Debug] Registered metallib" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("PagedAttentionPrimitive: Failed to register metallib '" + metallib_path + "': " + e.what());
    }

    // Get the kernel function
    std::string kernel_name = "paged_attn_kernel"; // Matches [[kernel]] name
     std::cerr << "[Debug] Getting kernel '" << kernel_name << "' from library '" << library_name << "'" << std::endl;
    auto kernel = d.get_kernel(kernel_name, library_name);
    if (!kernel) {
         throw std::runtime_error("PagedAttentionPrimitive: Failed to get kernel '" + kernel_name + "' from library '" + library_name + "'");
    }
    std::cerr << "[Debug] Got kernel object" << std::endl;

    // Get command encoder and set state
    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);
    std::cerr << "[Debug] Set pipeline state" << std::endl;

    // Set kernel arguments
    compute_encoder.set_input_array(q, 0);          // q_in at buffer(0)
    compute_encoder.set_input_array(kv_cache, 1);   // kv_in at buffer(1)
    compute_encoder.set_input_array(page_table, 2); // tbl_in at buffer(2)
    compute_encoder.set_output_array(out, 3);       // out_buf at buffer(3)
    std::cerr << "[Debug] Set kernel buffers" << std::endl;

    // Calculate grid/threadgroup dimensions (adjust based on actual kernel needs)
    size_t grid_dim_x = q.size(); // Example: based on total elements in q for stub
    if (grid_dim_x == 0) {
        std::cerr << "[Warning] PagedAttentionPrimitive: Input 'q' is empty, skipping dispatch." << std::endl;
        // Ensure output is zeroed or handled if needed for empty input
        // mx::fill(out, 0.0f, s); // Example if zeroing is desired
        return; // No kernel dispatch needed
    }
    MTL::Size grid_dims = MTL::Size(grid_dim_x, 1, 1);
    size_t tgp_size = kernel->maxTotalThreadsPerThreadgroup();
    if (tgp_size == 0 || tgp_size > grid_dim_x) {
         tgp_size = std::min((size_t)256, grid_dim_x); // Fallback/cap
    }
    if (tgp_size == 0) tgp_size = 1;
    MTL::Size group_dims = MTL::Size(tgp_size, 1, 1);
    std::cerr << "[Debug] Dispatching kernel: grid=" << grid_dim_x << ", group=" << tgp_size << std::endl;

    // Dispatch the kernel
    compute_encoder.dispatch_threads(grid_dims, group_dims);
    std::cerr << "[Debug] Kernel dispatched" << std::endl;
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
