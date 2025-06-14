// paged_attention.h.metal
// Optimized Paged Attention kernel for Metal.
//
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
// ============================================================================

#pragma once

#include <metal_stdlib>
#include "paged_attention_types.h"
#include "utils.h.metal"

using namespace metal;

constant bool USE_TWO_PASS [[function_constant(0)]];

template <typename T, int HEAD_DIM, int TOKENS_PER_PAGE>
[[kernel]] void pal_paged_attention(
    device const T*      queries_in             [[buffer(0)]],
    device const T*      k_cache_pool_in        [[buffer(1)]],
    device const T*      v_cache_pool_in        [[buffer(2)]],
    device const uint*   page_table_in          [[buffer(3)]],
    device const int*    context_lens_in        [[buffer(4)]],
    device T*            output_buffer          [[buffer(5)]],

    // --- Intermediate Buffers for Two-Pass ---
    device float*        max_logits_out         [[buffer(6), function_constant(USE_TWO_PASS)]],
    device float*        exp_sums_out           [[buffer(7), function_constant(USE_TWO_PASS)]],
    device T*            tmp_out                [[buffer(8), function_constant(USE_TWO_PASS)]],

    constant const PagedAttentionParams& params [[buffer(9)]],
    threadgroup uchar*   tg_mem                 [[threadgroup(0)]],
    uint3                tg_dim                 [[threads_per_threadgroup]],
    uint3                tg_pos_in_grid         [[threadgroup_position_in_grid]],
    uint                 local_idx_in_tg        [[thread_index_in_threadgroup]]
) {
    // Align our K cache to a MEMORY_ALIGNMENT_BYTES byte boundary for coalesced device memory access
    //
    // credit: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-paged-attn/src/metal/kernels/pagedattention.metal
    // for initial implementation and reference (they use x = 16 / sizeof(T))
    // =================================================================================
    // Physical memory access width (for coalesced device memory access)
    constexpr int ELEMENTS_PER_THREAD = MEMORY_ALIGNMENT_BYTES / sizeof(T);

    // --- 1. Identification ---
    uint seq_idx = tg_pos_in_grid.x;
    uint q_head_idx = tg_pos_in_grid.y;
    const int num_threads = tg_dim.x;

    // Determine the KV head this Q head should attend to.
    const int num_q_per_kv = params.num_q_heads / params.num_kv_heads;
    const int kv_head_id = q_head_idx / num_q_per_kv;

    // simd group and lane indices
    const int num_simd_groups = num_threads / SIMD_WIDTH;
    const int simdgroup_idx = local_idx_in_tg / SIMD_WIDTH;
    const int lane_idx = local_idx_in_tg % SIMD_WIDTH;

    // subgroup indices and lane offsets
    const int SUBGROUP_SIZE = MAX(SIMD_WIDTH / TOKENS_PER_PAGE, 1);
    const int NUM_SUBGROUPS = num_threads / SUBGROUP_SIZE;
    // where this subgroup is, "global" index
    const int subgroup_idx = local_idx_in_tg / SUBGROUP_SIZE;
    // where this thread is in the subgroup, "local" index
    const int subgroup_lane_offset = local_idx_in_tg % SUBGROUP_SIZE;

    // Logical computation vector width (for vectorized computation)
    constexpr int QK_VECTOR_WIDTH = MAX(MEMORY_ALIGNMENT_BYTES / (SUBGROUP_SIZE * sizeof(T)), 1);

    // how many vectors per thread
    const int num_vecs_per_thread = HEAD_DIM / (SUBGROUP_SIZE * QK_VECTOR_WIDTH);

    // vector types
    using VecQ = typename Vec<T, QK_VECTOR_WIDTH>::Type;
    using ScaledQVec = typename Vec<float, QK_VECTOR_WIDTH>::Type;
    using VecK = typename Vec<T, QK_VECTOR_WIDTH>::Type;

    // --- 2. Partition Threadgroup Memory ---
    threadgroup uchar* current_mem_ptr = tg_mem;

    // load the query vectors into shared memory
    threadgroup ScaledQVec* query_vectors = (threadgroup ScaledQVec*)current_mem_ptr;
    current_mem_ptr += sizeof(ScaledQVec) * SUBGROUP_SIZE * num_vecs_per_thread;
    #define QUERY_VEC(lane, vec) query_vectors[(lane) * num_vecs_per_thread + (vec)]

    threadgroup float* logits_tile = (threadgroup float*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (CHUNK_SIZE * sizeof(float)));

    // General-purpose scratchpad for reductions
    threadgroup float* reduction_scratchpad = (threadgroup float*)current_mem_ptr;
    current_mem_ptr = (threadgroup uchar*)ALIGN16(current_mem_ptr + (num_simd_groups * sizeof(float)));

    // --- 3. Load the Q Vector ---
    device const T* q_ptr = queries_in +
                            (seq_idx * params.num_q_heads * HEAD_DIM) +
                            (q_head_idx * HEAD_DIM);

    #pragma unroll
    for (int i = subgroup_idx; i < num_vecs_per_thread; i += NUM_SUBGROUPS) {
        int vec_offset = subgroup_lane_offset + i * SUBGROUP_SIZE;
        VecQ q_vec = ((device const VecQ*)q_ptr)[vec_offset];
        // Scale during load and store as float for precision in matmul.
        QUERY_VEC(subgroup_lane_offset, i) = ScaledQVec(q_vec) * params.inv_sqrt_head_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure the full Q-vector is in shared memory before use.

    // --- 4. Initialize Accumulators & Get Sequence Info ---
    const int context_len = context_lens_in[seq_idx];
    const int chunk_idx = tg_pos_in_grid.z; // we chunk by 512 tokens

    // Paged Attention Specific Variables
    const int total_pages = (context_len + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE;
    const int pages_per_chunk = CHUNK_SIZE / TOKENS_PER_PAGE;
    const int start_page_idx = USE_TWO_PASS ? (chunk_idx * pages_per_chunk) : 0;
    const int end_page_idx = USE_TWO_PASS ? MIN(start_page_idx + pages_per_chunk, total_pages) : total_pages;
    // token level variables
    const int start_token_idx = start_page_idx * TOKENS_PER_PAGE;
    const int end_token_idx = MIN(end_page_idx * TOKENS_PER_PAGE, context_len);
    const int num_tokens_in_chunk = end_token_idx - start_token_idx;
    // how many tokens per subgroup
    const int tokens_per_subgroup = (TOKENS_PER_PAGE + SIMD_WIDTH - 1) / SIMD_WIDTH;

    if (num_tokens_in_chunk <= 0) {
        // exit if the chunk is empty
        return;
    }

    // --- 5. Attention Loop ---
    // K-cache layout: [pages, kv_heads, head_size/ELEMENTS_PER_THREAD, page_size, ELEMENTS_PER_THREAD]
    const ulong k_head_stride = (ulong)HEAD_DIM * TOKENS_PER_PAGE;
    const ulong k_page_stride = (ulong)params.num_kv_heads * k_head_stride;

    // Online Softmax Statistics
    float max_score = -INFINITY;
    // --- 5.b Main Attention Loop ---
    for (
        int page_idx = start_page_idx + simdgroup_idx;
        page_idx < end_page_idx;
        page_idx += num_simd_groups
    ) {
        uint physical_page_id = page_table_in[seq_idx * params.max_logical_pages_per_seq + page_idx];

        #pragma unroll
        for (int i = 0; i < tokens_per_subgroup; ++i) {
            // 1. Calculate which token this subgroup works on.
            const int token_in_page = (subgroup_idx + i * SIMD_WIDTH) % TOKENS_PER_PAGE;
            const int global_token_idx = page_idx * TOKENS_PER_PAGE + token_in_page;

            // Boundary checks
            if (token_in_page >= TOKENS_PER_PAGE || global_token_idx >= context_len) {
                continue;
            }

            // 2. Load the K-vector for this token using the coalescing-aware layout.
            VecK k_vecs[num_vecs_per_thread];
            // Base pointer to the K data for this page and head.
            device const T* k_base_ptr = k_cache_pool_in +
                                        (ulong)physical_page_id * k_page_stride +
                                        (ulong)kv_head_id * k_head_stride;
            // Pointer to the start of the interleaved data for this specific token.
            device const T* k_token_ptr = k_base_ptr + token_in_page * ELEMENTS_PER_THREAD;

            #pragma unroll
            for (int j = 0; j < num_vecs_per_thread; j++) {
                // Calculate the logical vector index this thread is responsible for.
                const int logical_vec_idx = subgroup_lane_offset + j * SUBGROUP_SIZE;
                // Convert to an element offset from the start of the head dimension.
                const int total_elem_offset = logical_vec_idx * QK_VECTOR_WIDTH;

                // Decompose the element offset into two parts to navigate the physical memory layout.
                const int offset1 = total_elem_offset / ELEMENTS_PER_THREAD; // Which row of chunks.
                const int offset2 = total_elem_offset % ELEMENTS_PER_THREAD; // Offset within a chunk.

                // Calculate the final address and load the vector.
                k_vecs[j] = *reinterpret_cast<device const VecK*>(
                    k_token_ptr + (ulong)offset1 * TOKENS_PER_PAGE * ELEMENTS_PER_THREAD + offset2
                );
            }

            // 3. Compute the QK dot product
            float score = qk_dot_strided<ScaledQVec, VecK>(
                query_vectors,
                k_vecs,
                num_vecs_per_thread,
                subgroup_lane_offset,
                SUBGROUP_SIZE
            );

            if (subgroup_lane_offset == 0) {
                // only the leader thread of the subgroup writes the score
                logits_tile[global_token_idx - start_token_idx] = score;
                // update the global max score
                max_score = max(max_score, score);
            }
        } // end of tokens_per_subgroup loop
    } // end of main attention loop

    // --- 6. Global Max Score Reduction ---
    max_score = page_max(reduction_scratchpad, max_score, simdgroup_idx, lane_idx, num_simd_groups, SIMD_WIDTH, SUBGROUP_SIZE);

    // --- 7. Softmax Calculation ---
    float sum_exp = 0.0f; // thread local sum of exp(max(logits_tile - max_score, log_exp_min_clamp))
    // Each thread processes a unique subset of the tokens in the chunk in parallel.
    for (int i = local_idx_in_tg; i < num_tokens_in_chunk; i += num_threads) {
        // Read the raw score, subtract the global max, and compute exp.
        float val = exp(max(logits_tile[i] - max_score, params.log_exp_min_clamp));
        // Write the exponentiated value back to the logits tile in-place.
        logits_tile[i] = val;
        // Accumulate the value into the thread's local sum.
        sum_exp += val;
    }

    // 7b. Reduce the local sums to get the global sum for the chunk.
    sum_exp = page_sum(
        reduction_scratchpad,
        sum_exp,
        simdgroup_idx,
        lane_idx,
        num_simd_groups,
        SIMD_WIDTH
    );

    // 7c. Normalize the probabilities.
    const float inv_sum_exp = 1.0f / (sum_exp + 1e-6f);
    // 7d. Normalize the values in logits_tile.
    for (int i = local_idx_in_tg; i < num_tokens_in_chunk; i += num_threads) {
        logits_tile[i] *= inv_sum_exp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have finished writing the final probabilities before proceeding.

    // --- 8. V-Vector Aggregation ---
    // 8a. Define V-vector types and accumulator layout.
    // V-cache layout: [num_physical_pages, num_kv_heads, head_dim, tokens_per_page]
    const ulong v_head_stride = (ulong)HEAD_DIM * TOKENS_PER_PAGE;
    const ulong v_page_stride = (ulong)params.num_kv_heads * v_head_stride;

    constexpr int V_VECTOR_WIDTH = MIN(MEMORY_ALIGNMENT_BYTES / sizeof(T), TOKENS_PER_PAGE);
    using V_vec = typename Vec<T, V_VECTOR_WIDTH>::Type;
    using P_vec = typename Vec<float, V_VECTOR_WIDTH>::Type; // Probability vector

    // How many elements of the output vector does each SIMD group handle in one pass?
    constexpr int ELEMENTS_PER_SIMD= SIMD_WIDTH / (TOKENS_PER_PAGE / V_VECTOR_WIDTH);
    // How many passes does each thread need to cover its assigned elements?
    constexpr int PASSES_PER_THREAD = (HEAD_DIM + ELEMENTS_PER_SIMD - 1) / ELEMENTS_PER_SIMD;
    // Per-thread accumulator registers, initialized to zero.
    float acc[PASSES_PER_THREAD] = {0.0f};

    // 8b. The V-aggregation loop.
    for (
        int page_idx = start_page_idx + simdgroup_idx;
        page_idx < end_page_idx;
        page_idx += num_simd_groups
    ) {
        uint physical_page_id = page_table_in[seq_idx * params.max_logical_pages_per_seq + page_idx];
        device const T* v_page_ptr = v_cache_pool_in +
                                    (ulong)physical_page_id * v_page_stride +
                                    (ulong)kv_head_id * v_head_stride;

        // This logic distributes token processing within a SIMD group for V-aggregation.
        const int token_in_page = (lane_idx % (TOKENS_PER_PAGE / V_VECTOR_WIDTH)) * V_VECTOR_WIDTH;
        const int global_token_idx = page_idx * TOKENS_PER_PAGE + token_in_page;

        if (token_in_page >= TOKENS_PER_PAGE || global_token_idx >= context_len) {
            continue;
        }

        // Load the vector of probabilities for the tokens this thread will process.
        P_vec probs_vec = *reinterpret_cast<threadgroup P_vec*>(
            logits_tile + (global_token_idx - start_token_idx)
        );

        #pragma unroll
        for (int i = 0; i < PASSES_PER_THREAD; ++i) {
            // Which element of the output vector is this thread working on?
            const int element_idx = (lane_idx / (TOKENS_PER_PAGE / V_VECTOR_WIDTH)) + i * ELEMENTS_PER_SIMD;

            if (element_idx < HEAD_DIM) {
                // Load the V-vector for this element and these tokens.
                const int offset = element_idx * TOKENS_PER_PAGE + token_in_page;
                V_vec v_vec = *reinterpret_cast<device const V_vec*>(v_page_ptr + offset);

                // Boundary check for the last page
                if (page_idx == total_pages - 1) {
                    thread T* v_vec_ptr = reinterpret_cast<thread T*>(&v_vec);
                    #pragma unroll
                    for (int j = 0; j < V_VECTOR_WIDTH; j++) {
                        if (global_token_idx + j >= context_len) {
                            v_vec_ptr[j] = T(0.0f);
                        }
                    }
                }

                // Accumulate the weighted sum.
                acc[i] += dot(probs_vec, v_vec);
            }
        }
    }

    // --- 9. Final Output Reduction ---

    // 9a. Intra-SIMD Reduction
    // Sum the values horizontally across the tokens processed by the SIMD group.
    #pragma unroll
    for (int i = 0; i < PASSES_PER_THREAD; ++i) {
        float partial_sum = acc[i];
        #pragma unroll
        for (int mask = (TOKENS_PER_PAGE / V_VECTOR_WIDTH) / 2; mask >= 1; mask /= 2) {
            partial_sum += simd_shuffle_xor(partial_sum, mask);
        }
        acc[i] = partial_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // THREADGROUP BARRIER REASON: Ensure all threads have finished accumulating their V vectors.

    // 9b. Inter-SIMD (Cross-Warp) Reduction
    // The logits_tile is now repurposed as `out_smem`.
    threadgroup float* out_smem = logits_tile;

    #pragma unroll
    for (int i = num_simd_groups; i > 1; i /= 2) {
        int mid = i / 2;

        // The upper half of the active SIMD groups write their results to shared memory.
        if (simdgroup_idx >= mid && simdgroup_idx < i) {
            threadgroup float* dst = &out_smem[(simdgroup_idx - mid) * HEAD_DIM];
            #pragma unroll
            for (int j = 0; j < PASSES_PER_THREAD; ++j) {
                const int element_idx = (lane_idx / (TOKENS_PER_PAGE / V_VECTOR_WIDTH)) + j * ELEMENTS_PER_SIMD;
                // Only the leader for each element writes.
                if (element_idx < HEAD_DIM && (lane_idx % (TOKENS_PER_PAGE / V_VECTOR_WIDTH) == 0)) {
                    dst[element_idx] = acc[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // THREADGROUP BARRIER REASON: Ensure all threads have finished writing their results to shared memory.

        // The lower half of the active SIMD groups read and update their accumulators.
        if (simdgroup_idx < mid) {
            const threadgroup float* src = &out_smem[simdgroup_idx * HEAD_DIM];
            #pragma unroll
            for (int j = 0; j < PASSES_PER_THREAD; ++j) {
                const int element_idx = (lane_idx / (TOKENS_PER_PAGE / V_VECTOR_WIDTH)) + j * ELEMENTS_PER_SIMD;
                if (element_idx < HEAD_DIM && (lane_idx % (TOKENS_PER_PAGE / V_VECTOR_WIDTH) == 0)) {
                    acc[j] += src[element_idx];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // THREADGROUP BARRIER REASON: Ensure all threads have finished reading and updating their accumulators.
    }
    // --- 10. Finalization and Output ---
    // 10a. Two-Pass Finalization
    if (USE_TWO_PASS) {
        const int max_chunks = (params.max_logical_pages_per_seq * TOKENS_PER_PAGE + CHUNK_SIZE - 1) / CHUNK_SIZE;
        // --- Two-Pass (Pass 1) Finalization ---
        if (local_idx_in_tg == 0) {
            // The output index is based on which sequence, head, and chunk we are.
            const int out_idx = (seq_idx * params.num_q_heads * max_chunks) +
                                (q_head_idx * max_chunks) +
                                chunk_idx;
            max_logits_out[out_idx] = max_score;
            exp_sums_out[out_idx] = sum_exp;
        }

        if (simdgroup_idx == 0) {
            // Base offset to the start of this sequence and head's data.
            const ulong base_offset = ((ulong)seq_idx * params.num_q_heads + q_head_idx) * max_chunks * HEAD_DIM;
            // Chunk-specific offset to the start of the current chunk's head_dim vector.
            const ulong chunk_offset = (ulong)chunk_idx * HEAD_DIM;
            device T* tmp_out_ptr = tmp_out + base_offset + chunk_offset;

            #pragma unroll
            for (int i = 0; i < PASSES_PER_THREAD; ++i) {
                const int element_idx = (lane_idx / (TOKENS_PER_PAGE / V_VECTOR_WIDTH)) + i * ELEMENTS_PER_SIMD;
                // Only the leader thread for each element writes its computed value.
                if (element_idx < HEAD_DIM && (lane_idx % (TOKENS_PER_PAGE / V_VECTOR_WIDTH) == 0)) {
                    tmp_out_ptr[element_idx] = T(acc[i]);
                }
            }
        }
        return;
    }

    // 10b. Single-Pass Finalization
    if (simdgroup_idx == 0) {
        // The output pointer is based on which sequence and head we are.
        device T* out_ptr = output_buffer +
                            (seq_idx * params.num_q_heads * HEAD_DIM) +
                            (q_head_idx * HEAD_DIM);

        #pragma unroll
        for (int i = 0; i < PASSES_PER_THREAD; ++i) {
            const int element_idx = (lane_idx / (TOKENS_PER_PAGE / V_VECTOR_WIDTH)) + i * ELEMENTS_PER_SIMD;
            // Only the leader thread for each element writes its computed value.
            if (element_idx < HEAD_DIM && (lane_idx % (TOKENS_PER_PAGE / V_VECTOR_WIDTH) == 0)) {
                out_ptr[element_idx] = T(acc[i]);
            }
        }
    }

} // end of pal_paged_attention
