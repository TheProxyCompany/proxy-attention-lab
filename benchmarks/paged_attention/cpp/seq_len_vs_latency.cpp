// Copyright 2025 The Proxy Company. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
//
// Google Benchmark for PAL C++ Operations

#include "pal_core/paged_attention_primitive.hpp"
#include <benchmark/benchmark.h>
#include <cmath>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/array.h>
#include <mlx/random.h>
#include <mlx/ops.h>

#include <mlx/fast.h>
#include <spdlog/spdlog.h>

#include "pal_core/ops.hpp"
#include "pal_core/metal/metal_loader.hpp"

namespace mx = mlx::core;

// Static initializer to set spdlog level for benchmarks
struct BenchmarkSpdlogInitializer {
    BenchmarkSpdlogInitializer() {
        spdlog::set_level(spdlog::level::off);
    }
};
// This object will be constructed before main(), ensuring the log level is set
static BenchmarkSpdlogInitializer global_benchmark_spdlog_initializer;

// Define baseline configuration for benchmarks - matching the Python version exactly
// Gemma 3 Model Config, 2048 tokens
struct BaselineConfig {
    int batch_size = 1;
    int seq_len = 2048;  // tokens
    int num_q_heads = 32;
    int num_kv_heads = 16;
    int head_dim = 128;
    int tokens_per_page = 56;
    mx::Dtype dtype = mx::float16;
};

// Decode-specific batch size
const int DECODE_BATCH_SIZE = 1;

// Helper function to create causal mask (for SDPA)
mx::array create_causal_mask(int seq_len, mx::Dtype dtype) {
    auto ones = mx::full({seq_len, seq_len}, 1.0f, dtype);
    auto mask_val = std::numeric_limits<float>::lowest();
    auto upper_triangular = mx::triu(ones, 1);
    mx::array causal_mask = upper_triangular * mask_val;
    return mx::astype(causal_mask, dtype);
}

// Helper function to create the page table
mx::array create_page_table(int num_sequences_in_batch, int num_logical_pages_per_seq) {
    const int total_entries = num_sequences_in_batch * num_logical_pages_per_seq;
    std::vector<uint32_t> pt_data(total_entries);

    for (int b_idx = 0; b_idx < num_sequences_in_batch; ++b_idx) {
        for (int l_idx = 0; l_idx < num_logical_pages_per_seq; ++l_idx) {
            pt_data[b_idx * num_logical_pages_per_seq + l_idx] =
                static_cast<uint32_t>(b_idx * num_logical_pages_per_seq + l_idx);
        }
    }

    // Construct the array from raw data and explicitly provide its shape.
    return mx::array(
        pt_data.data(),
        {num_sequences_in_batch, num_logical_pages_per_seq},
        mx::uint32
    );
}

// C++ PAL Decode benchmark function
static void BM_PAL_DecodeLatencyVsHistoryLen(benchmark::State& state) {
    // Create test parameters from baseline with specified history length
    BaselineConfig params;
    params.batch_size = DECODE_BATCH_SIZE;
    int history_len = state.range(0); // Use the history length from benchmark args

    // Extract params to local variables for clarity
    int batch_size = params.batch_size;
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    params.tokens_per_page = pal::cpp::PagedAttentionPrimitive::get_optimal_page_size();
    int tokens_per_page = params.tokens_per_page;

    // Setup input tensors for decode scenario
    int num_tokens = batch_size;  // One token per sequence for decode
    int num_logical_pages_per_seq = (history_len + tokens_per_page - 1) / tokens_per_page;
    int num_total_physical_pages = batch_size * num_logical_pages_per_seq;

    // Create query tensor with shape [num_tokens, num_q_heads, head_dim]
    // For decode, this represents a single new token per sequence
    mx::array queries = mx::random::normal(
        {num_tokens, num_q_heads, head_dim},
        dtype
    );

    // Use the new helper functions to get correct cache shapes
    // K-cache shape: [num_total_physical_pages, num_kv_heads, head_dim / elements_per_thread, tokens_per_page, elements_per_thread]
    mx::Shape k_cache_shape = pal::cpp::PagedAttentionPrimitive::get_k_cache_shape(
        num_total_physical_pages, num_kv_heads, head_dim, tokens_per_page, dtype
    );
    mx::array k_cache_pool = mx::random::normal(k_cache_shape, dtype);

    // V-cache shape: [num_total_physical_pages, num_kv_heads, head_dim, tokens_per_page]
    mx::Shape v_cache_shape = pal::cpp::PagedAttentionPrimitive::get_v_cache_shape(
        num_total_physical_pages, num_kv_heads, head_dim, tokens_per_page, dtype
    );
    mx::array v_cache_pool = mx::random::normal(v_cache_shape, dtype);

    // Create page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    mx::array page_table = create_page_table(batch_size, num_logical_pages_per_seq);

    // Set sequence length for each batch item to the history length
    mx::array sequence_lengths = mx::full({batch_size}, history_len, mx::int32);

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = pal::cpp::paged_attention(
            queries,
            k_cache_pool,
            v_cache_pool,
            page_table,
            sequence_lengths
        );
        out.eval();
    }
}

// MLX SDPA Decode benchmark function
static void BM_MLX_SDPA_DecodeLatencyVsHistoryLen(benchmark::State& state) {
    // Create parameters with history length
    BaselineConfig params;
    params.batch_size = DECODE_BATCH_SIZE;
    int history_len = state.range(0); // Use the history length from benchmark args

    // Extract params to local variables for clarity
    int batch_size = params.batch_size;
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors for decode (single token attending to history)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create query tensor representing a single new token per sequence
    // Shape: [batch_size, num_q_heads, 1, head_dim]
    mx::array queries = mx::random::normal(
        {batch_size, num_q_heads, 1, head_dim}, dtype
    );

    // Create keys and values tensors representing the history
    // Shape: [batch_size, num_kv_heads, history_len, head_dim]
    mx::array keys = mx::random::normal(
        {batch_size, num_kv_heads, history_len, head_dim}, dtype
    );

    mx::array values = mx::random::normal(
        {batch_size, num_kv_heads, history_len, head_dim}, dtype
    );

    // Create a mask of zeros (allows full attention) with shape [1, history_len]
    mx::array mask = mx::zeros({1, history_len}, dtype);

    // Log tensor shapes to verify scaling with history_len
    spdlog::info("DECODE SDPA - history_len: {}", history_len);
    spdlog::info("  queries shape: [{}, {}, {}, {}]", batch_size, num_q_heads, 1, head_dim);
    spdlog::info("  keys shape: [{}, {}, {}, {}]", batch_size, num_kv_heads, history_len, head_dim);
    spdlog::info("  values shape: [{}, {}, {}, {}]", batch_size, num_kv_heads, history_len, head_dim);
    spdlog::info("  mask shape: [{}, {}]", 1, history_len);

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            "array",
            {mask}
        );
        out.eval();
    }
}

const int REPETITIONS = 10; // magic number
const int ITERATIONS = 100; // magic number

BENCHMARK(BM_PAL_DecodeLatencyVsHistoryLen)
   ->Arg(64)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(128)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(256)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(1024)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(2048)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(4096)->Iterations(ITERATIONS)->Repetitions(REPETITIONS);

BENCHMARK(BM_MLX_SDPA_DecodeLatencyVsHistoryLen)
   ->Arg(64)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(128)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(256)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(1024)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(2048)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
   ->Arg(4096)->Iterations(ITERATIONS)->Repetitions(REPETITIONS);

BENCHMARK_MAIN();
