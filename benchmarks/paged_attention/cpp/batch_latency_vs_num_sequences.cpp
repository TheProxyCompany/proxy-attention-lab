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
// Google Benchmark for PAL Batch Operations (Both Fused and Two-Pass)

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

// Define baseline configuration for benchmarks - Gemma 3 Model Config
struct BaselineConfig {
    int batch_size = 1;
    int num_q_heads = 32;
    int num_kv_heads = 16;
    int head_dim = 128;
    int tokens_per_page = 56;  // Will be overridden by optimal size
    mx::Dtype dtype = mx::float16;
};

// Helper function to create causal mask (for SDPA prefill)
static mx::array create_causal_mask(int seq_len, mx::Dtype dtype) {
    auto ones = mx::full({seq_len, seq_len}, 1.0f, dtype);
    auto mask_val = std::numeric_limits<float>::lowest();
    auto upper_triangular = mx::triu(ones, 1);
    mx::array causal_mask = upper_triangular * mask_val;
    return mx::astype(causal_mask, dtype);
}

// Helper function to create the page table for batch decode
static mx::array create_page_table(int num_sequences_in_batch, int num_logical_pages_per_seq) {
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

// Helper function to create sequence-to-query mapping for prefill
static mx::array create_query_to_seq_map_prefill(int num_sequences, int seq_len) {
    // Create a query-to-sequence map that matches the python version:
    // query_to_seq_map = mx.repeat(mx.arange(num_sequences, dtype=mx.int32), repeats=seq_len)
    std::vector<int32_t> q_to_seq_data(num_sequences * seq_len);

    for (int b_idx = 0; b_idx < num_sequences; ++b_idx) {
        for (int s_idx = 0; s_idx < seq_len; ++s_idx) {
            q_to_seq_data[b_idx * seq_len + s_idx] = b_idx;
        }
    }

    return mx::array(q_to_seq_data.data(), {num_sequences * seq_len}, mx::int32);
}

// Helper function to create query token offsets for prefill
static mx::array create_query_token_offset_prefill(int num_sequences, int seq_len) {
    // Create query token offsets that match the python version:
    // query_token_offset = mx.tile(mx.arange(1, seq_len + 1, dtype=mx.int32), num_sequences)
    std::vector<int32_t> offset_data(num_sequences * seq_len);

    for (int b_idx = 0; b_idx < num_sequences; ++b_idx) {
        for (int s_idx = 0; s_idx < seq_len; ++s_idx) {
            offset_data[b_idx * seq_len + s_idx] = s_idx + 1; // 1-indexed
        }
    }

    return mx::array(offset_data.data(), {num_sequences * seq_len}, mx::int32);
}


// Benchmark function that varies history length (H)
static void BM_PAL_DecodeBatchLatencyVsHistoryLength(benchmark::State& state) {
    // Extract benchmark parameters
    BaselineConfig params;
    int num_sequences = state.range(0); // N: number of sequences in batch
    int history_length = state.range(1); // H: history length for each sequence

    // Extract params to local variables for clarity
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Get optimal tile size
    auto info = pal::cpp::PagedAttentionPrimitive::get_optimal_tile_size_and_thread_info();

    params.tokens_per_page = std::get<0>(info);
    int tokens_per_page = params.tokens_per_page;

    // Setup input tensors for decode scenario
    int num_tokens = num_sequences;  // One token per sequence for decode
    int num_logical_pages_per_seq = (history_length + tokens_per_page - 1) / tokens_per_page;
    int num_total_physical_pages = num_sequences * num_logical_pages_per_seq;

    // Create query tensor with shape [num_tokens, num_q_heads, head_dim]
    // For decode, this represents one new token per sequence
    mx::array queries = mx::random::normal(
        {num_tokens, num_q_heads, head_dim},
        dtype
    );

    // K/V cache pools sized for the history: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    mx::array k_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim},
        dtype
    );
    mx::array v_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim},
        dtype
    );

    // Create page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    mx::array page_table = create_page_table(num_sequences, num_logical_pages_per_seq);

    // Set sequence length for each batch item to the history length
    mx::array sequence_lengths = mx::full({num_sequences}, history_length, mx::int32);

    // Create query-to-sequence mapping for decode (just sequence indices)
    mx::array query_to_seq_map = mx::arange(num_sequences, mx::int32);

    // Create query token offsets for decode (position after history)
    std::vector<int32_t> offset_data(num_sequences);
    for (int i = 0; i < num_sequences; ++i) {
        offset_data[i] = history_length + 1; // Position after history
    }
    mx::array query_token_offset = mx::array(offset_data.data(), {num_sequences}, mx::int32);

    queries = mx::contiguous(queries);
    k_cache_pool = mx::contiguous(k_cache_pool);
    v_cache_pool = mx::contiguous(v_cache_pool);
    page_table = mx::contiguous(page_table);
    sequence_lengths = mx::contiguous(sequence_lengths);
    query_to_seq_map = mx::contiguous(query_to_seq_map);
    query_token_offset = mx::contiguous(query_token_offset);

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
            true /* use_fused_kernel */
        );
        out.eval();
    }
}

// PAL Two-Pass Prefill Batch benchmark function varying sequence length
static void BM_PAL_TwoPass_PrefillBatchLatencyVsSeqLen(benchmark::State& state) {
    // Extract benchmark parameters
    BaselineConfig params;
    int seq_len = state.range(0);        // L: sequence length for each prefill
    int num_sequences = state.range(1);  // N: number of sequences in batch

    // Extract params to local variables for clarity
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Get optimal tile size
    auto info = pal::cpp::PagedAttentionPrimitive::get_optimal_tile_size_and_thread_info();

    params.tokens_per_page = std::get<0>(info);
    int tokens_per_page = params.tokens_per_page;

    // Setup input tensors for prefill scenario
    int num_tokens = num_sequences * seq_len;  // Total tokens across all sequences
    int num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) / tokens_per_page;
    int num_total_physical_pages = num_sequences * num_logical_pages_per_seq;

    // Create query tensor with shape [num_tokens, num_q_heads, head_dim]
    mx::array queries = mx::random::normal(
        {num_tokens, num_q_heads, head_dim},
        dtype
    );

    // K/V cache pools: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    mx::array k_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim},
        dtype
    );
    mx::array v_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim},
        dtype
    );

    // Create page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    mx::array page_table = create_page_table(num_sequences, num_logical_pages_per_seq);

    // Set sequence length for each batch item
    mx::array sequence_lengths = mx::full({num_sequences}, seq_len, mx::int32);

    // Create query-to-sequence mapping for prefill
    mx::array query_to_seq_map = create_query_to_seq_map_prefill(num_sequences, seq_len);

    // Create query token offsets for prefill
    mx::array query_token_offset = create_query_token_offset_prefill(num_sequences, seq_len);

    queries = mx::contiguous(queries);
    k_cache_pool = mx::contiguous(k_cache_pool);
    v_cache_pool = mx::contiguous(v_cache_pool);
    page_table = mx::contiguous(page_table);
    sequence_lengths = mx::contiguous(sequence_lengths);
    query_to_seq_map = mx::contiguous(query_to_seq_map);
    query_token_offset = mx::contiguous(query_token_offset);

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
            false /* use_fused_kernel - two-pass */
        );
        out.eval();
    }
}

// MLX SDPA Batch Decode benchmark function varying num sequences
static void BM_MLX_SDPA_DecodeBatchLatencyVsNumSequences(benchmark::State& state) {
    // Extract benchmark parameters
    BaselineConfig params;
    int num_sequences = state.range(0);  // N: number of sequences in batch
    int history_length = state.range(1); // H: history length for each sequence

    // Extract params to local variables for clarity
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors for decode (single token attending to history)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create query tensor representing one new token per sequence
    // Shape: [num_sequences, num_q_heads, 1, head_dim]
    mx::array queries = mx::random::normal(
        {num_sequences, num_q_heads, 1, head_dim}, dtype
    );

    // Create keys and values tensors representing the history
    // Shape: [num_sequences, num_kv_heads, history_length, head_dim]
    mx::array keys = mx::random::normal(
        {num_sequences, num_kv_heads, history_length, head_dim}, dtype
    );

    mx::array values = mx::random::normal(
        {num_sequences, num_kv_heads, history_length, head_dim}, dtype
    );

    // Create a mask of zeros (allows full attention) with shape [1, history_length]
    mx::array mask = mx::zeros({1, history_length}, dtype);

    queries.eval();
    keys.eval();
    values.eval();
    mask.eval();

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

// MLX SDPA Batch Decode benchmark function varying history length
static void BM_MLX_SDPA_DecodeBatchLatencyVsHistoryLength(benchmark::State& state) {
    // Extract benchmark parameters
    BaselineConfig params;
    int num_sequences = state.range(0); // N: number of sequences in batch
    int history_length = state.range(1); // H: history length for each sequence

    // Extract params to local variables for clarity
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors for decode (single token attending to history)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create query tensor representing one new token per sequence
    // Shape: [num_sequences, num_q_heads, 1, head_dim]
    mx::array queries = mx::random::normal(
        {num_sequences, num_q_heads, 1, head_dim}, dtype
    );

    // Create keys and values tensors representing the history
    // Shape: [num_sequences, num_kv_heads, history_length, head_dim]
    mx::array keys = mx::random::normal(
        {num_sequences, num_kv_heads, history_length, head_dim}, dtype
    );

    mx::array values = mx::random::normal(
        {num_sequences, num_kv_heads, history_length, head_dim}, dtype
    );

    // Create a mask of zeros (allows full attention) with shape [1, history_length]
    mx::array mask = mx::zeros({1, history_length}, dtype);

    queries.eval();
    keys.eval();
    values.eval();
    mask.eval();

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

// MLX SDPA Batch Prefill benchmark function varying num sequences
static void BM_MLX_SDPA_PrefillBatchLatencyVsNumSequences(benchmark::State& state) {
    // Extract benchmark parameters
    BaselineConfig params;
    int num_sequences = state.range(0);  // N: number of sequences in batch
    int seq_len = state.range(1);        // L: sequence length for each prefill

    // Extract params to local variables for clarity
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors for prefill (full sequences)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create query, key, value tensors for prefill
    // Shape: [num_sequences, num_q_heads, seq_len, head_dim]
    mx::array queries = mx::random::normal(
        {num_sequences, num_q_heads, seq_len, head_dim}, dtype
    );

    mx::array keys = mx::random::normal(
        {num_sequences, num_kv_heads, seq_len, head_dim}, dtype
    );

    mx::array values = mx::random::normal(
        {num_sequences, num_kv_heads, seq_len, head_dim}, dtype
    );

    // Create causal mask for prefill
    mx::array causal_mask = create_causal_mask(seq_len, dtype);

    queries.eval();
    keys.eval();
    values.eval();
    causal_mask.eval();

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            "array",
            {causal_mask}
        );
        out.eval();
    }
}

// MLX SDPA Batch Prefill benchmark function varying sequence length
static void BM_MLX_SDPA_PrefillBatchLatencyVsSeqLen(benchmark::State& state) {
    // Extract benchmark parameters
    BaselineConfig params;
    int seq_len = state.range(0);        // L: sequence length for each prefill
    int num_sequences = state.range(1);  // N: number of sequences in batch

    // Extract params to local variables for clarity
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors for prefill (full sequences)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create query, key, value tensors for prefill
    // Shape: [num_sequences, num_q_heads, seq_len, head_dim]
    mx::array queries = mx::random::normal(
        {num_sequences, num_q_heads, seq_len, head_dim}, dtype
    );

    mx::array keys = mx::random::normal(
        {num_sequences, num_kv_heads, seq_len, head_dim}, dtype
    );

    mx::array values = mx::random::normal(
        {num_sequences, num_kv_heads, seq_len, head_dim}, dtype
    );

    // Create causal mask for prefill
    mx::array causal_mask = create_causal_mask(seq_len, dtype);

    queries.eval();
    keys.eval();
    values.eval();
    causal_mask.eval();

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            "array",
            {causal_mask}
        );
        out.eval();
    }
}

const int REPETITIONS = 3;
const int ITERATIONS = 3;

// PAL: Benchmark varying H (history length) for each batch size
BENCHMARK(BM_PAL_DecodeBatchLatencyVsHistoryLength)
    ->Args({4, 128})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({4, 256})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({4, 1024})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({4, 4096})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({16, 128})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({16, 256})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({16, 1024})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({16, 4096})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 128})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 256})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 1024})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 4096})->Repetitions(REPETITIONS)->Iterations(ITERATIONS);

// MLX: Benchmark varying H (history length) for each batch size
BENCHMARK(BM_MLX_SDPA_DecodeBatchLatencyVsHistoryLength)
    ->Args({4, 128})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({4, 256})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({4, 1024})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({4, 4096})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({16, 128})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({16, 256})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({16, 1024})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({16, 4096})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 128})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 256})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 1024})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 4096})->Repetitions(REPETITIONS)->Iterations(ITERATIONS);

// PAL Two-Pass: Benchmark varying sequence length for smaller batch sizes
BENCHMARK(BM_PAL_TwoPass_PrefillBatchLatencyVsSeqLen)
    ->Args({32, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({128, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({256, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({512, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({32, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({128, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({256, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({512, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS);

// MLX SDPA Prefill: Benchmark varying sequence length for smaller batch sizes
BENCHMARK(BM_MLX_SDPA_PrefillBatchLatencyVsSeqLen)
    ->Args({32, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({128, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({256, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({512, 4})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({32, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({64, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({128, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({256, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS)
    ->Args({512, 16})->Repetitions(REPETITIONS)->Iterations(ITERATIONS);
