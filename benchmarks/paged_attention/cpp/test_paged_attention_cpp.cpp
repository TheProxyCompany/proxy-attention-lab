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
#include "pal_core/metal/metal_loader.hpp"

namespace mx = mlx::core;

// Static initializer to set spdlog level for benchmarks
struct BenchmarkSpdlogInitializer {
    BenchmarkSpdlogInitializer() {
        // Set default log level for benchmarks to debug to see GPU copy logs
        spdlog::set_level(spdlog::level::debug);
        spdlog::info("PAL C++ Benchmarks: spdlog level set to 'debug' for debugging GPU copies.");
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
    int head_dim = 128; // bottleneck dim
    int tokens_per_page = 58;
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

    k_cache_pool.eval();
    v_cache_pool.eval();
    queries.eval();
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
            true // use prefill mode for this benchmark
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

    // Create causal mask that scales with sequence length
    mx::array causal_mask = create_causal_mask(seq_len, dtype);

    // Log tensor shapes to verify scaling with seq_len
    spdlog::info("PREFILL SDPA - seq_len: {}", seq_len);
    spdlog::info("  queries shape: [{}, {}, {}, {}]", batch_size, num_q_heads, seq_len, head_dim);
    spdlog::info("  keys shape: [{}, {}, {}, {}]", batch_size, num_kv_heads, seq_len, head_dim);
    spdlog::info("  values shape: [{}, {}, {}, {}]", batch_size, num_kv_heads, seq_len, head_dim);
    spdlog::info("  causal_mask shape: [{}, {}]", seq_len, seq_len);

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


// only need this one time to register the metal lib
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
        true
    ); // use prefill mode
    warm.eval();                  // wait for GPU

    mx::array warm_decode = pal::cpp::paged_attention(
        queries, k_cache_pool, v_cache_pool,
        page_table, sequence_lengths,
        query_to_seq_map, query_token_offset,
        false
    ); // use decode mode
    warm_decode.eval();                  // wait for GPU
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

    k_cache_pool.eval();
    v_cache_pool.eval();
    queries.eval();
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
            false // use decode mode for this benchmark
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

const int REPETITIONS = 20;
const int ITERATIONS = 5;

// BENCHMARK(BM_PAL_LatencyVsSeqLen)
//    ->Arg(64)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)->Setup(BM_PAL_LatencyVsSeqLen_Setup)
//    ->Arg(256)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(512)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(1024)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(2048)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(4096)->Iterations(ITERATIONS)->Repetitions(REPETITIONS);

// BENCHMARK(BM_MLX_SDPA_LatencyVsSeqLen)
//    ->Arg(64)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(256)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(512)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(1024)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(2048)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(4096)->Iterations(ITERATIONS)->Repetitions(REPETITIONS);

// BENCHMARK(BM_PAL_DecodeLatencyVsHistoryLen)
//    ->Arg(64)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(128)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(256)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(1024)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(2048)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(4096)->Iterations(ITERATIONS)->Repetitions(REPETITIONS);

// BENCHMARK(BM_MLX_SDPA_DecodeLatencyVsHistoryLen)
//    ->Arg(64)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(128)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(256)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(1024)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(2048)->Iterations(ITERATIONS)->Repetitions(REPETITIONS)
//    ->Arg(4096)->Iterations(ITERATIONS)->Repetitions(REPETITIONS);

static void test_paged_attention_dummy_data_flow() { // Renamed for clarity
    spdlog::info("Running paged attention dummy data flow test...");

    // Test configuration to achieve num_active_batch_logical_pages = 3
    int batch_size = 1;
    int seq_len = 3;      // This many query tokens, and length of sequence for page calculation
    int tokens_per_page = 1;  // Each token gets its own page, so 3 tokens -> 3 pages

    int num_q_heads = 1;
    int num_kv_heads = 1; // Dummy kernel writes as if kv_head_idx = 0
    int head_dim = 4;     // Must be >= 4 for the dummy output check
    mx::Dtype dtype = mx::float32; // Using float32 for easier verification of sums

    // This setup will result in:
    // num_logical_pages_per_seq = (seq_len + tokens_per_page - 1) / tokens_per_page
    //                           = (3 + 1 - 1) / 1 = 3
    // num_active_batch_logical_pages = batch_size * num_logical_pages_per_seq = 1 * 3 = 3
    // So, flat_work_item_idx in Pass 1 will be 0, 1, 2.

    int query_token_count = batch_size * seq_len; // Total query tokens for the 'queries' array
    int num_physical_pages = batch_size * ((seq_len + tokens_per_page - 1) / tokens_per_page);
    if (num_physical_pages == 0) num_physical_pages = 1; // Ensure at least one physical page for cache

    // Create dummy tensors (content doesn't matter for these specific dummy kernels)
    mx::array queries = mx::zeros({query_token_count, num_q_heads, head_dim}, dtype);
    mx::array k_cache_pool = mx::zeros({num_physical_pages, tokens_per_page, num_kv_heads, head_dim}, dtype);
    mx::array v_cache_pool = mx::zeros({num_physical_pages, tokens_per_page, num_kv_heads, head_dim}, dtype);

    // Metadata arrays
    mx::array page_table = create_page_table(batch_size, (seq_len + tokens_per_page - 1) / tokens_per_page);
    mx::array sequence_lengths = mx::full({batch_size}, seq_len, mx::int32);
    mx::array query_to_seq_map = create_query_to_seq_map(batch_size, seq_len);
    mx::array query_token_offset = create_query_token_offset(batch_size, seq_len); // Content doesn't matter for dummy kernels

    // Evaluate all inputs
    queries.eval(); k_cache_pool.eval(); v_cache_pool.eval();
    page_table.eval(); sequence_lengths.eval();
    query_to_seq_map.eval(); query_token_offset.eval();

    // Run paged attention
    mx::array out = pal::cpp::paged_attention(
        queries, k_cache_pool, v_cache_pool, page_table,
        sequence_lengths, query_to_seq_map, query_token_offset,
        true // prefill mode
    );
    out.eval(); // Crucial to ensure computation completes

    // Verification
    // Output shape from paged_attention for 3D queries [QTC, NQH, HD] is [QTC * NQH, HD]
    // Here, QTC=3, NQH=1, HD=4. So, out shape is [3, 4].
    // The dummy Pass 2 kernel writes to output_item_idx = 0 (for DUMMY_QUERY_TOKEN_IDX=0, DUMMY_Q_HEAD_IDX=0).
    // So we check out[0][0], out[0][1], out[0][2], out[0][3].

    // Expected values based on dummy Metal kernels:
    // flat_work_item_idx will be 0, 1, 2.
    // accumulated_m_val = (0+1.0f) + (1+1.0f) + (2+1.0f) = 1.0 + 2.0 + 3.0 = 6.0f
    // accumulated_s_val = (0+1.0f)*100 + (1+1.0f)*100 + (2+1.0f)*100 = 100 + 200 + 300 = 600.0f
    // accumulated_o_val_elem0 = (half)(0+0.5f) + (half)(1+0.5f) + (half)(2+0.5f) = 0.5h + 1.5h + 2.5h = 4.5h
    // accumulated_o_val_elem1 = (half)(0+0.5f) + (half)(1+0.5f) + (half)(2+0.5f) = 0.5h + 1.5h + 2.5h = 4.5h
    float expected_val0 = 6.0f;
    float expected_val1 = 600.0f;
    float expected_val2 = 4.5f; // This will be float due to accumulation in Pass 2
    float expected_val3 = 4.5f; // Same pattern for additional elements

    // Accessing elements from the 2D 'out' array
    // Ensure out array is not empty and has the expected shape
    if (out.size() == 0 || out.ndim() != 2 || out.shape(0) < 1 || out.shape(1) < head_dim) {
        spdlog::error("Dummy data test FAILED: Output array is empty or has unexpected shape: [{}, {}]", out.shape(0), out.shape(1));
        throw std::runtime_error("Test failed due to output array shape.");
    }

    // Use mx::slice (free function) to extract individual elements, then .item<T>() for scalar extraction
    // For a 2D array, slice from [start_row, start_col] to [end_row, end_col]
    mx::array elem0 = mx::slice(out, {0, 0}, {1, 1});
    mx::array elem1 = mx::slice(out, {0, 1}, {1, 2});
    mx::array elem2 = mx::slice(out, {0, 2}, {1, 3});
    mx::array elem3 = mx::slice(out, {0, 3}, {1, 4});

    float val0 = elem0.item<float>();
    float val1 = elem1.item<float>();
    float val2 = elem2.item<float>();
    float val3 = elem3.item<float>();

    spdlog::info("Dummy data flow test results:");
    spdlog::info("  Output array shape: [{}, {}]", out.shape(0), out.shape(1));
    spdlog::info("  out[0][0] (accumulated_m_val) = {} (expected: {})", val0, expected_val0);
    spdlog::info("  out[0][1] (accumulated_s_val) = {} (expected: {})", val1, expected_val1);
    spdlog::info("  out[0][2] (accumulated_o_val_elem0) = {} (expected: {})", val2, expected_val2);
    spdlog::info("  out[0][3] (accumulated_o_val_elem1) = {} (expected: {})", val3, expected_val3);

    const float tolerance = 1e-3f; // Tolerance for float comparisons
    bool test_passed = true;

    if (std::abs(val0 - expected_val0) > tolerance) {
        spdlog::error("Test FAILED: out[0][0] mismatch.");
        test_passed = false;
    }
    if (std::abs(val1 - expected_val1) > tolerance) {
        spdlog::error("Test FAILED: out[0][1] mismatch.");
        test_passed = false;
    }
    if (std::abs(val2 - expected_val2) > tolerance) {
        spdlog::error("Test FAILED: out[0][2] mismatch.");
        test_passed = false;
    }
    if (std::abs(val3 - expected_val3) > tolerance) {
        spdlog::error("Test FAILED: out[0][3] mismatch.");
        test_passed = false;
    }

    if (test_passed) {
        spdlog::info("Dummy data flow test PASSED!");
    } else {
        spdlog::error("Dummy data flow test FAILED!");
        // Optionally, throw an exception to make the benchmark fail clearly
        // throw std::runtime_error("Dummy data flow test failed verification.");
    }
}

// Call the test function before benchmarks
struct DummyDataTestRunner {
    DummyDataTestRunner() {
        test_paged_attention_dummy_data_flow();
    }
};
static DummyDataTestRunner dummy_test_runner;

BENCHMARK_MAIN();
