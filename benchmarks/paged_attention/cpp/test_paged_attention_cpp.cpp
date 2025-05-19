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
#include "pal_core/metal_loader.hpp"

namespace mx = mlx::core;

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

// Define baseline configuration for benchmarks - matching the Python version exactly
// Gemma 3 Model Config, 2048 tokens
struct BaselineConfig {
    int batch_size = 1;
    int seq_len = 2048;  // tokens
    int num_q_heads = 32;
    int num_kv_heads = 16;
    int head_dim = 128;
    int tokens_per_page = 64;
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

// Helper function to create sequence-to-query mapping
mx::array create_query_to_seq_map(int batch_size, int seq_len) {
    // Create a query-to-sequence map that matches the python version:
    // query_to_seq_map = mx.repeat(mx.arange(batch_size, dtype=mx.int32), repeats=seq_len)
    std::vector<int32_t> q_to_seq_data(batch_size * seq_len);

    for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (int s_idx = 0; s_idx < seq_len; ++s_idx) {
            q_to_seq_data[b_idx * seq_len + s_idx] = b_idx;
        }
    }

    return mx::array(q_to_seq_data.data(), {batch_size * seq_len}, mx::int32);
}

// Helper function to create query token offsets
mx::array create_query_token_offset(int batch_size, int seq_len) {
    // Create query token offsets that match the python version:
    // query_token_offset = mx.tile(mx.arange(1, seq_len + 1, dtype=mx.int32), batch_size)
    std::vector<int32_t> offset_data(batch_size * seq_len);

    for (int b_idx = 0; b_idx < batch_size; ++b_idx) {
        for (int s_idx = 0; s_idx < seq_len; ++s_idx) {
            offset_data[b_idx * seq_len + s_idx] = s_idx + 1; // 1-indexed
        }
    }

    return mx::array(offset_data.data(), {batch_size * seq_len}, mx::int32);
}

static void BM_PAL_LatencyVsSeqLen(benchmark::State& state) {
    // Create test parameters from baseline with specified sequence length
    BaselineConfig params;
    params.seq_len = state.range(0); // Use the sequence length from benchmark args

    // Extract params to local variables for clarity
    int batch_size = params.batch_size;
    int seq_len = params.seq_len;
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    int tokens_per_page = params.tokens_per_page;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors
    int num_tokens = batch_size * seq_len;
    int num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) / tokens_per_page;
    int num_total_physical_pages = batch_size * num_logical_pages_per_seq;

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
    mx::array page_table = create_page_table(batch_size, num_logical_pages_per_seq);

    // Set sequence length for each batch item: [num_sequences_in_batch]
    mx::array sequence_lengths = mx::full({batch_size}, seq_len, mx::int32);

    // Create query-to-sequence mapping: [num_tokens]
    mx::array query_to_seq_map = create_query_to_seq_map(batch_size, seq_len);

    // Create query token offsets: [num_tokens]
    mx::array query_token_offset = create_query_token_offset(batch_size, seq_len);

    // ----------------------------------------------------
    //  Warm-up (compile shader, register Metal lib, etc.)
    // ----------------------------------------------------
    mx::array warm = pal::cpp::paged_attention(
        queries, k_cache_pool, v_cache_pool,
        page_table, sequence_lengths,
        query_to_seq_map, query_token_offset,
        mx::Device::gpu);
    warm.eval();                  // wait for GPU

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
        out.eval();
    }
}

static void BM_MLX_SDPA_LatencyVsSeqLen(benchmark::State& state) {
    // Create parameters with the fixed batch size and specified sequence length
    BaselineConfig params;
    params.seq_len = state.range(0); // Use the sequence length from benchmark args

    // Extract params to local variables for clarity
    int batch_size = params.batch_size;
    int seq_len = params.seq_len;
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors (matching Python benchmark)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create input tensors with shapes matching the Python benchmark
    mx::array queries = mx::random::normal(
        {batch_size, num_q_heads, seq_len, head_dim}, dtype
    );

    mx::array keys = mx::random::normal(
        {batch_size, num_kv_heads, seq_len, head_dim}, dtype
    );

    mx::array values = mx::random::normal(
        {batch_size, num_kv_heads, seq_len, head_dim}, dtype
    );

    // Create causal mask
    mx::array causal_mask = create_causal_mask(seq_len, dtype);

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            "array",
            {causal_mask},
            mx::Device::gpu
        );
        out.eval();
    }
}


void BM_PAL_LatencyVsSeqLen_Setup(const ::benchmark::State& state) {
    // Create test parameters from baseline with specified sequence length
    BaselineConfig params;
    params.seq_len = 16;

    // Extract params to local variables for clarity
    int batch_size = params.batch_size;
    int seq_len = params.seq_len;
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    int tokens_per_page = params.tokens_per_page;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors
    int num_tokens = batch_size * seq_len;
    int num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) / tokens_per_page;
    int num_total_physical_pages = batch_size * num_logical_pages_per_seq;

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
    mx::array page_table = create_page_table(batch_size, num_logical_pages_per_seq);

    // Set sequence length for each batch item: [num_sequences_in_batch]
    mx::array sequence_lengths = mx::full({batch_size}, seq_len, mx::int32);

    // Create query-to-sequence mapping: [num_tokens]
    mx::array query_to_seq_map = create_query_to_seq_map(batch_size, seq_len);

    // Create query token offsets: [num_tokens]
    mx::array query_token_offset = create_query_token_offset(batch_size, seq_len);

    // ----------------------------------------------------
    //  Warm-up (compile shader, register Metal lib, etc.)
    // ----------------------------------------------------
    mx::array warm = pal::cpp::paged_attention(
        queries, k_cache_pool, v_cache_pool,
        page_table, sequence_lengths,
        query_to_seq_map, query_token_offset,
        mx::Device::gpu);
    warm.eval();                  // wait for GPU
}

// Register the benchmarks with all sequence lengths from Python benchmark
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
    int tokens_per_page = params.tokens_per_page;
    mx::Dtype dtype = params.dtype;

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
    mx::array page_table = create_page_table(batch_size, num_logical_pages_per_seq);

    // Set sequence length for each batch item to the history length
    mx::array sequence_lengths = mx::full({batch_size}, history_len, mx::int32);

    // Create query-to-sequence mapping for decode (just sequence indices)
    mx::array query_to_seq_map = mx::arange(batch_size, mx::int32);

    // Create query token offsets for decode (position after history)
    std::vector<int32_t> offset_data(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        offset_data[i] = history_len + 1; // Position after history
    }
    mx::array query_token_offset = mx::array(offset_data.data(), {batch_size}, mx::int32);

    // ----------------------------------------------------
    //  Warm-up (compile shader, register Metal lib, etc.)
    // ----------------------------------------------------
    mx::array warm = pal::cpp::paged_attention(
        queries, k_cache_pool, v_cache_pool,
        page_table, sequence_lengths,
        query_to_seq_map, query_token_offset,
        mx::Device::gpu);
    warm.eval();

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

    // No mask needed for decode - query attends to all history items

    // Create a mask of zeros (allows full attention) with shape [1, history_len]
    mx::array mask = mx::zeros({1, history_len}, dtype);

    // Main benchmark loop
    for (auto _ : state) {
        mx::array out = mx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            "array",
            {mask}, // Provide mask that allows full attention
            mx::Device::gpu
        );
        out.eval();
    }
}

BENCHMARK(BM_PAL_LatencyVsSeqLen)
    ->Arg(64)->Iterations(1)->Repetitions(1)->Setup(BM_PAL_LatencyVsSeqLen_Setup)
    // ->Arg(128)
    // ->Arg(256)
    // ->Arg(512)
    ->Arg(1024)->Iterations(1)->Repetitions(1);
    // ->Arg(2048)
    // ->Arg(4096);

BENCHMARK(BM_MLX_SDPA_LatencyVsSeqLen)
    ->Arg(64)->Iterations(1)->Repetitions(1)
    // ->Arg(128)
    // ->Arg(256)
    // ->Arg(512)
    ->Arg(1024)->Iterations(1)->Repetitions(1);
    // ->Arg(2048)
    // ->Arg(4096);

// Setup function for the PAL decode benchmark
void BM_PAL_DecodeLatencyVsHistoryLen_Setup(const ::benchmark::State& state) {
    // Create test parameters for a small decode test
    BaselineConfig params;
    params.batch_size = DECODE_BATCH_SIZE;
    int history_len = 64; // Small history for quick warmup

    // Extract params to local variables for clarity
    int batch_size = params.batch_size;
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    int tokens_per_page = params.tokens_per_page;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors for decode scenario
    int num_tokens = batch_size;  // One token per sequence for decode
    int num_logical_pages_per_seq = (history_len + tokens_per_page - 1) / tokens_per_page;
    int num_total_physical_pages = batch_size * num_logical_pages_per_seq;

    // Create query tensor with shape [num_tokens, num_q_heads, head_dim]
    mx::array queries = mx::random::normal({num_tokens, num_q_heads, head_dim}, dtype);

    // K/V cache pools: [num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim]
    mx::array k_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim}, dtype
    );
    mx::array v_cache_pool = mx::random::normal(
        {num_total_physical_pages, tokens_per_page, num_kv_heads, head_dim}, dtype
    );

    // Create page table: [num_sequences_in_batch, num_logical_pages_per_seq]
    mx::array page_table = create_page_table(batch_size, num_logical_pages_per_seq);

    // Set sequence length for each batch item to the history length
    mx::array sequence_lengths = mx::full({batch_size}, history_len, mx::int32);

    // Create query-to-sequence mapping for decode (just sequence indices)
    mx::array query_to_seq_map = mx::arange(batch_size, mx::int32);

    // Create query token offsets for decode (position after history)
    std::vector<int32_t> offset_data(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        offset_data[i] = history_len + 1; // Position after history
    }
    mx::array query_token_offset = mx::array(offset_data.data(), {batch_size}, mx::int32);

    // Warmup to compile shaders, register Metal lib, etc.
    mx::array warm = pal::cpp::paged_attention(
        queries, k_cache_pool, v_cache_pool,
        page_table, sequence_lengths,
        query_to_seq_map, query_token_offset,
        mx::Device::gpu);
    warm.eval();
}

// Register the decode benchmarks with history lengths
BENCHMARK(BM_PAL_DecodeLatencyVsHistoryLen)
    ->Arg(1024)->Iterations(1)->Repetitions(1)->Setup(BM_PAL_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(2048)->Iterations(1)->Repetitions(1)->Setup(BM_PAL_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(4096)->Iterations(1)->Repetitions(1)->Setup(BM_PAL_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(8192)->Iterations(1)->Repetitions(1)->Setup(BM_PAL_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(16384)->Iterations(1)->Repetitions(1)->Setup(BM_PAL_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(32768)->Iterations(1)->Repetitions(1)->Setup(BM_PAL_DecodeLatencyVsHistoryLen_Setup);

// Setup function for the MLX SDPA decode benchmark
void BM_MLX_SDPA_DecodeLatencyVsHistoryLen_Setup(const ::benchmark::State& state) {
    // Create test parameters for a small decode test
    BaselineConfig params;
    params.batch_size = DECODE_BATCH_SIZE;
    int history_len = 64; // Small history for quick warmup

    // Extract params to local variables
    int batch_size = params.batch_size;
    int num_q_heads = params.num_q_heads;
    int num_kv_heads = params.num_kv_heads;
    int head_dim = params.head_dim;
    mx::Dtype dtype = params.dtype;

    // Setup input tensors for decode scenario
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create query tensor for a single token per sequence
    mx::array queries = mx::random::normal({batch_size, num_q_heads, 1, head_dim}, dtype);

    // Create keys and values tensors for history
    mx::array keys = mx::random::normal({batch_size, num_kv_heads, history_len, head_dim}, dtype);
    mx::array values = mx::random::normal({batch_size, num_kv_heads, history_len, head_dim}, dtype);

    // Create an empty mask that allows full attention (zeros)
    mx::array mask = mx::zeros({1, history_len}, dtype);

    // Warmup to compile and cache any kernel
    mx::array warm = mx::fast::scaled_dot_product_attention(
        queries, keys, values, scale, "array", {mask}, mx::Device::gpu
    );
    warm.eval();
}

BENCHMARK(BM_MLX_SDPA_DecodeLatencyVsHistoryLen)
    ->Arg(1024)->Iterations(1)->Repetitions(1)->Setup(BM_MLX_SDPA_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(2048)->Iterations(1)->Repetitions(1)->Setup(BM_MLX_SDPA_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(4096)->Iterations(1)->Repetitions(1)->Setup(BM_MLX_SDPA_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(8192)->Iterations(1)->Repetitions(1)->Setup(BM_MLX_SDPA_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(16384)->Iterations(1)->Repetitions(1)->Setup(BM_MLX_SDPA_DecodeLatencyVsHistoryLen_Setup)
    ->Arg(32768)->Iterations(1)->Repetitions(1)->Setup(BM_MLX_SDPA_DecodeLatencyVsHistoryLen_Setup);

BENCHMARK_MAIN();
