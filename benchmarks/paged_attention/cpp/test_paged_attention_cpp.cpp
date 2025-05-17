// Copyright 2024 The Proxy Company. All Rights Reserved.
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

namespace mx = mlx::core;

// Helper struct for SDPA benchmark inputs (using a tuple instead of a struct to avoid constructor issues)
using SDPABenchmarkInputs = std::tuple<mx::array, mx::array, mx::array, float, mx::array>;

// Helper function to set up SDPA benchmark inputs
SDPABenchmarkInputs setup_sdpa_benchmark_inputs_cpp(
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    mx::Dtype dtype
) {
    // Compute scale factor (1.0 / sqrt(head_dim))
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create input tensors with appropriate shapes
    mx::array queries = mx::random::normal(
        {batch_size, num_q_heads, seq_len, head_dim},
        dtype
    );

    mx::array keys = mx::random::normal(
        {batch_size, num_kv_heads, seq_len, head_dim},
        dtype
    );

    mx::array values = mx::random::normal(
        {batch_size, num_kv_heads, seq_len, head_dim},
        dtype
    );

    // Create causal mask (using MLX's triu and full functions)
    // Create a full ones matrix and make it upper triangular for causal masking
    auto ones = mx::full({seq_len, seq_len}, 1.0f, dtype);
    auto mask_val = std::numeric_limits<float>::lowest();
    auto upper_triangular = mx::triu(ones, 1);
    mx::array causal_mask = upper_triangular * mask_val;

    // Ensure the causal_mask is of the same dtype as queries/keys/values.
    // The multiplication above (e.g., float16 array * float scalar) promotes the result to float32.
    // We need to cast it back to the original dtype (e.g., float16) for the SDPA kernel.
    causal_mask = mx::astype(causal_mask, dtype);

    // Ensure tensors are computed before returning
    queries.eval();
    keys.eval();
    values.eval();
    causal_mask.eval();

    // Return tuple with all inputs
    return {queries, keys, values, scale, causal_mask};
}

// Static initializer to set spdlog level for benchmarks
struct BenchmarkSpdlogInitializer {
    BenchmarkSpdlogInitializer() {
        // Set default log level for benchmarks to warning to reduce noise
        spdlog::set_level(spdlog::level::warn);
        spdlog::info("PAL C++ Benchmarks: spdlog level set to 'warn'. Debug/trace messages from pal_core_lib will be suppressed.");
    }
};
// This object will be constructed before main(), ensuring the log level is set
static BenchmarkSpdlogInitializer global_benchmark_spdlog_initializer;

namespace mx = mlx::core;

// Default configuration for benchmarks
const int DEFAULT_HEAD_DIM = 128;
const int DEFAULT_NUM_Q_HEADS = 1;
const int DEFAULT_NUM_KV_HEADS = 1;
const int DEFAULT_SEQ_LEN = 128;
const int DEFAULT_TOKENS_PER_PAGE = 64;
const int DEFAULT_NUM_SEQUENCES_IN_BATCH = 1;
const int DEFAULT_NUM_QUERY_ITEMS = 64;  // For DEFAULT_NUM_Q_HEADS=1, this means 64 tokens

// Default configuration for SDPA benchmarks
const int SDPA_DEFAULT_BATCH_SIZE = 64;
const int SDPA_DEFAULT_NUM_Q_HEADS = 1;
const int SDPA_DEFAULT_NUM_KV_HEADS = 1;
const int SDPA_DEFAULT_HEAD_DIM = 128;
const int SDPA_DEFAULT_SEQ_LEN = 128;

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

// Helper function to create the query to sequence map
mx::array create_query_to_seq_map(int num_tokens, int num_sequences_in_batch) {
    std::vector<int32_t> q_to_seq_data(num_tokens);

    if (num_sequences_in_batch == 1) {
        std::fill(q_to_seq_data.begin(), q_to_seq_data.end(), 0);
    } else {
        const int tokens_per_seq = num_tokens / num_sequences_in_batch;
        for (int s_idx = 0; s_idx < num_sequences_in_batch; ++s_idx) {
            for (int t_idx = 0; t_idx < tokens_per_seq; ++t_idx) {
                q_to_seq_data[s_idx * tokens_per_seq + t_idx] = s_idx;
            }
        }
    }

    // Construct the array from raw data and explicitly provide its shape.
    return mx::array(
        q_to_seq_data.data(),
        {num_tokens},
        mx::int32
    );
}

static void BM_PAL_LatencyVsSeqLen(benchmark::State& state) {
    // Extract parameters from state
    const int num_query_items = state.range(0);
    const int seq_len = state.range(1);

    // Fixed parameters for this benchmark
    const int num_q_heads = DEFAULT_NUM_Q_HEADS;
    const int num_kv_heads = DEFAULT_NUM_KV_HEADS;
    const int head_dim = DEFAULT_HEAD_DIM;
    const int tokens_per_page = DEFAULT_TOKENS_PER_PAGE;
    const int num_sequences_in_batch = DEFAULT_NUM_SEQUENCES_IN_BATCH;

    // Derived parameters
    const int num_tokens = num_query_items / num_q_heads;
    // Ensure valid configuration
    if (num_query_items % num_q_heads != 0) {
        state.SkipWithError("num_query_items must be divisible by num_q_heads");
        return;
    }

    const int num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) / tokens_per_page;
    const int num_total_physical_pages = num_sequences_in_batch * num_logical_pages_per_seq;

    // Create input tensors
    // Queries: [num_tokens, num_q_heads, head_dim]
    mx::array queries = mx::random::normal({num_tokens, num_q_heads, head_dim}, mx::float16);

    // K/V cache pools: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    mx::array k_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim}, mx::float16);

    mx::array v_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim}, mx::float16);

    // Page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    mx::array page_table = create_page_table(num_sequences_in_batch, num_logical_pages_per_seq);

    // Sequence lengths: [num_sequences_in_batch]
    mx::array sequence_lengths = mx::full({num_sequences_in_batch}, seq_len, mx::int32);

    // Query to sequence map: [num_tokens]
    mx::array query_to_seq_map = create_query_to_seq_map(num_tokens, num_sequences_in_batch);

    // Query token offset: [num_tokens]
    mx::array query_token_offset = mx::full({num_tokens}, seq_len, mx::int32);

    // Ensure tensors are ready before benchmarking
    queries.eval();
    k_cache_pool.eval();
    v_cache_pool.eval();
    page_table.eval();
    sequence_lengths.eval();
    query_to_seq_map.eval();
    query_token_offset.eval();

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = pal::cpp::paged_attention(
            queries,
            k_cache_pool,
            v_cache_pool,
            page_table,
            sequence_lengths,
            query_to_seq_map,
            query_token_offset,
            mx::Device::gpu
        );
        out.eval();  // Ensure GPU computation completes
    }

    // Set the number of bytes processed
    size_t bytes_processed = static_cast<size_t>(num_query_items) * head_dim * sizeof(float);
    state.SetBytesProcessed(static_cast<long long>(bytes_processed) * state.iterations());
}

static void BM_PAL_LatencyVsHeadDim(benchmark::State& state) {
    // Extract parameters from state
    const int num_query_items = state.range(0);
    const int head_dim = state.range(1);

    // Fixed parameters for this benchmark
    const int num_q_heads = DEFAULT_NUM_Q_HEADS;
    const int num_kv_heads = DEFAULT_NUM_KV_HEADS;
    const int seq_len = DEFAULT_SEQ_LEN;
    const int tokens_per_page = DEFAULT_TOKENS_PER_PAGE;
    const int num_sequences_in_batch = DEFAULT_NUM_SEQUENCES_IN_BATCH;

    // Derived parameters
    const int num_tokens = num_query_items / num_q_heads;
    // Ensure valid configuration
    if (num_query_items % num_q_heads != 0) {
        state.SkipWithError("num_query_items must be divisible by num_q_heads");
        return;
    }

    const int num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) / tokens_per_page;
    const int num_total_physical_pages = num_sequences_in_batch * num_logical_pages_per_seq;

    // Create input tensors
    // Queries: [num_tokens, num_q_heads, head_dim]
    mx::array queries = mx::random::normal({num_tokens, num_q_heads, head_dim}, mx::float16);

    // K/V cache pools: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    mx::array k_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim}, mx::float16);

    mx::array v_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim}, mx::float16);

    // Page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    mx::array page_table = create_page_table(num_sequences_in_batch, num_logical_pages_per_seq);

    // Sequence lengths: [num_sequences_in_batch]
    mx::array sequence_lengths = mx::full({num_sequences_in_batch}, seq_len, mx::int32);

    // Query to sequence map: [num_tokens]
    mx::array query_to_seq_map = create_query_to_seq_map(num_tokens, num_sequences_in_batch);

    // Query token offset: [num_tokens]
    mx::array query_token_offset = mx::full({num_tokens}, seq_len, mx::int32);

    // Ensure tensors are ready before benchmarking
    queries.eval();
    k_cache_pool.eval();
    v_cache_pool.eval();
    page_table.eval();
    sequence_lengths.eval();
    query_to_seq_map.eval();
    query_token_offset.eval();

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = pal::cpp::paged_attention(
            queries,
            k_cache_pool,
            v_cache_pool,
            page_table,
            sequence_lengths,
            query_to_seq_map,
            query_token_offset,
            mx::Device::gpu
        );
        out.eval();  // Ensure GPU computation completes
    }

    // Set the number of bytes processed
    size_t bytes_processed = static_cast<size_t>(num_query_items) * head_dim * sizeof(float);
    state.SetBytesProcessed(static_cast<long long>(bytes_processed) * state.iterations());
}

static void BM_PAL_LatencyVsNumItems(benchmark::State& state) {
    // Extract parameters from state
    const int num_query_items = state.range(0);

    // Fixed parameters for this benchmark
    const int num_q_heads = DEFAULT_NUM_Q_HEADS;
    const int num_kv_heads = DEFAULT_NUM_KV_HEADS;
    const int head_dim = DEFAULT_HEAD_DIM;
    const int seq_len = DEFAULT_SEQ_LEN;
    const int tokens_per_page = DEFAULT_TOKENS_PER_PAGE;
    const int num_sequences_in_batch = DEFAULT_NUM_SEQUENCES_IN_BATCH;

    // Derived parameters
    const int num_tokens = num_query_items / num_q_heads;
    // Ensure valid configuration
    if (num_query_items % num_q_heads != 0) {
        state.SkipWithError("num_query_items must be divisible by num_q_heads");
        return;
    }

    const int num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) / tokens_per_page;
    const int num_total_physical_pages = num_sequences_in_batch * num_logical_pages_per_seq;

    // Create input tensors
    // Queries: [num_tokens, num_q_heads, head_dim]
    mx::array queries = mx::random::normal({num_tokens, num_q_heads, head_dim}, mx::float16);

    // K/V cache pools: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    mx::array k_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim}, mx::float16);

    mx::array v_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim}, mx::float16);

    // Page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    mx::array page_table = create_page_table(num_sequences_in_batch, num_logical_pages_per_seq);

    // Sequence lengths: [num_sequences_in_batch]
    mx::array sequence_lengths = mx::full({num_sequences_in_batch}, seq_len, mx::int32);

    // Query to sequence map: [num_tokens]
    mx::array query_to_seq_map = create_query_to_seq_map(num_tokens, num_sequences_in_batch);

    // Query token offset: [num_tokens]
    mx::array query_token_offset = mx::full({num_tokens}, seq_len, mx::int32);

    // Ensure tensors are ready before benchmarking
    queries.eval();
    k_cache_pool.eval();
    v_cache_pool.eval();
    page_table.eval();
    sequence_lengths.eval();
    query_to_seq_map.eval();
    query_token_offset.eval();

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = pal::cpp::paged_attention(
            queries,
            k_cache_pool,
            v_cache_pool,
            page_table,
            sequence_lengths,
            query_to_seq_map,
            query_token_offset,
            mx::Device::gpu
        );
        out.eval();  // Ensure GPU computation completes
    }

    // Set the number of items processed
    state.SetItemsProcessed(static_cast<long long>(num_query_items) * state.iterations());
}

// Register the benchmarks
// Use a constant number of query vectors (8192) across the sweep so that
// results are comparable with the Python benchmarks.
BENCHMARK(BM_PAL_LatencyVsSeqLen)
    ->Args({8192, 64})
    ->Args({8192, 128})
    ->Args({8192, 256})
    ->Args({8192, 512})
    ->Args({8192, 1024})
    ->Args({8192, 2048});

BENCHMARK(BM_PAL_LatencyVsHeadDim)
    ->Args({64*1, 64})
    ->Args({64*1, 128})
    ->Args({64*1, 192})
    ->Args({64*1, 256});

BENCHMARK(BM_PAL_LatencyVsNumItems)
    ->RangeMultiplier(2)
    ->Range(32, 512);

// SDPA Benchmark Functions

static void BM_SDPA_LatencyVsSeqLen(benchmark::State& state) {
    // Extract parameters from state
    const int batch_size = static_cast<int>(state.range(0));
    const int seq_len = static_cast<int>(state.range(1));

    // Fixed parameters for this benchmark
    const int num_q_heads = SDPA_DEFAULT_NUM_Q_HEADS;
    const int num_kv_heads = SDPA_DEFAULT_NUM_KV_HEADS;
    const int head_dim = SDPA_DEFAULT_HEAD_DIM;

    // Setup input tensors using the helper function
    auto [queries, keys, values, scale, causal_mask] = setup_sdpa_benchmark_inputs_cpp(
        batch_size, num_q_heads, num_kv_heads, head_dim, seq_len, mx::float16
    );

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            "array",  // Using explicit mask array
            {causal_mask},
            mx::Device::gpu
        );
        out.eval();  // Ensure GPU computation completes
    }

    // Set the number of items processed (batch_size * num_q_heads * seq_len per iteration)
    state.SetItemsProcessed(static_cast<long long>(batch_size * num_q_heads * seq_len) * state.iterations());
}

static void BM_SDPA_LatencyVsHeadDim(benchmark::State& state) {
    // Extract parameters from state
    const int batch_size = static_cast<int>(state.range(0));
    const int head_dim = static_cast<int>(state.range(1));

    // Fixed parameters for this benchmark
    const int num_q_heads = SDPA_DEFAULT_NUM_Q_HEADS;
    const int num_kv_heads = SDPA_DEFAULT_NUM_KV_HEADS;
    const int seq_len = SDPA_DEFAULT_SEQ_LEN;

    // Setup input tensors using the helper function
    auto [queries, keys, values, scale, causal_mask] = setup_sdpa_benchmark_inputs_cpp(
        batch_size, num_q_heads, num_kv_heads, head_dim, seq_len, mx::float16
    );

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            "array",  // Using explicit mask array
            {causal_mask},
            mx::Device::gpu
        );
        out.eval();  // Ensure GPU computation completes
    }

    // Set the number of bytes processed (batch_size * num_q_heads * seq_len * head_dim * sizeof(float))
    size_t bytes_processed = static_cast<size_t>(batch_size * num_q_heads * seq_len) * head_dim * sizeof(float);
    state.SetBytesProcessed(static_cast<long long>(bytes_processed) * state.iterations());
}

static void BM_SDPA_LatencyVsNumItems(benchmark::State& state) {
    // Extract parameters from state - for SDPA this is batch_size
    const int batch_size = static_cast<int>(state.range(0));

    // Fixed parameters for this benchmark
    const int num_q_heads = SDPA_DEFAULT_NUM_Q_HEADS;
    const int num_kv_heads = SDPA_DEFAULT_NUM_KV_HEADS;
    const int head_dim = SDPA_DEFAULT_HEAD_DIM;
    const int seq_len = SDPA_DEFAULT_SEQ_LEN;

    // Setup input tensors using the helper function
    auto [queries, keys, values, scale, causal_mask] = setup_sdpa_benchmark_inputs_cpp(
        batch_size, num_q_heads, num_kv_heads, head_dim, seq_len, mx::float16
    );

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            "array",  // Using explicit mask array
            {causal_mask},
            mx::Device::gpu
        );
        out.eval();  // Ensure GPU computation completes
    }

    // Set the number of items processed (batch_size * num_q_heads * seq_len per iteration)
    // This matches the "num_query_items" metric in the PAL benchmarks
    state.SetItemsProcessed(static_cast<long long>(batch_size * num_q_heads * seq_len) * state.iterations());
}

// Register the SDPA benchmarks
// Adjust batch size so that batch_size * seq_len equals 8192
// (num_q_heads is fixed at 1). This mirrors the PAL benchmark.
BENCHMARK(BM_SDPA_LatencyVsSeqLen)
    ->Args({128, 64})   // 128 * 64 = 8192
    ->Args({64, 128})   // 64 * 128 = 8192
    ->Args({32, 256})   // 32 * 256 = 8192
    ->Args({16, 512})   // 16 * 512 = 8192
    ->Args({8, 1024})   // 8 * 1024 = 8192
    ->Args({4, 2048});  // 4 * 2048 = 8192

BENCHMARK(BM_SDPA_LatencyVsHeadDim)
    ->Args({SDPA_DEFAULT_BATCH_SIZE, 64})
    ->Args({SDPA_DEFAULT_BATCH_SIZE, 128})
    ->Args({SDPA_DEFAULT_BATCH_SIZE, 192})
    ->Args({SDPA_DEFAULT_BATCH_SIZE, 256});

BENCHMARK(BM_SDPA_LatencyVsNumItems)
    ->RangeMultiplier(2)
    ->Range(32, 512);

// Model Configuration Benchmarks have been removed

// Main function
BENCHMARK_MAIN();
