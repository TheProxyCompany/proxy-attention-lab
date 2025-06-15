#pragma once

#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>
#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/metal.h>
#include <mlx/backend/metal/utils.h>
#include <vector>
#include "kernels/paged_attention_types.h"

namespace mx = mlx::core;

namespace pal::cpp {

class PagedAttentionPrefillPrimitive : public mx::UnaryPrimitive {
public:
    explicit PagedAttentionPrefillPrimitive(
        mx::StreamOrDevice stream_or_device,
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int tokens_per_page
    );

  /**
   * @brief Evaluates the primitive on CPU.
   *
   * @param inputs Vector of input arrays
   * @param out Output array to store the result
   */
  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override;


    void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;

    void print(std::ostream& os) override;

    // is_equivalent MUST now check q_tile_size, as it changes the kernel.
    bool is_equivalent(const mx::Primitive& other) const override;

    std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override;

private:
    int num_q_heads_;
    int num_kv_heads_;
    int head_dim_;
    int tokens_per_page_;
    int q_tile_size_; // Store the tile size
};

} // namespace pal::cpp
