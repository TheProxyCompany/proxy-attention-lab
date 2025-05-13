#pragma once

#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/array.h>
#include <vector>
#include <string>
#include <optional>
#include "mlx/utils.h"


namespace mx = mlx::core;

namespace pal::cpp {

// Define the custom primitive class inheriting from UnaryPrimitive
class PagedAttentionPrimitive : public mx::UnaryPrimitive {
public:
    // Constructor now accepts key parameters that determine kernel behavior
    explicit PagedAttentionPrimitive(
        mx::StreamOrDevice stream,
        int num_q_heads = 0,
        int num_kv_heads = 0,
        int head_dim = 0,
        int tokens_per_page = 0
    );

    // Evaluation on CPU (can be stubbed for now)
    void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override;

    // Evaluation on GPU (this will contain the kernel launch logic)
    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;

    // Print method for debugging/graph visualization
    void print(std::ostream& os) override;

    // Equivalence check (important for graph caching/optimization)
    // Compares stored parameters that affect kernel behavior
    bool is_equivalent(const mx::Primitive& other) const override;

    // Output shape calculation based on input shapes
    std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override;

private:
    // Store key parameters that affect kernel behavior
    int num_q_heads_;
    int num_kv_heads_;
    int head_dim_;
    int tokens_per_page_;

    // Gradient definitions (stubbed for now, can be implemented later if needed)
    std::vector<mx::array> vjp(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<mx::array>& outputs) override;

    std::vector<mx::array> jvp(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) override;

    // Vmap definition (stubbed for now)
    std::pair<std::vector<mx::array>, std::vector<int>> vmap(
        const std::vector<mx::array>& inputs,
        const std::vector<int>& axes) override;

};

} // namespace pal::cpp
