// paged_attention_fused.h.metal
// Metal shader header for paged attention operations with tiled V accumulation.
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
#include "pal_types.h.metal"

using namespace metal;

inline float warp_sum(float v, uint simd_size) {
    for (uint off = simd_size >> 1; off > 0; off >>= 1)
        v += simd_shuffle_down(v, off);
    return v;
}
inline float warp_max(float v, uint simd_size) {
    for (uint off = simd_size >> 1; off > 0; off >>= 1)
        v = max(v, simd_shuffle_down(v, off));
    return v;
}

template <typename T>
[[kernel]] void paged_attn_fused_kernel(
    device      const T*    queries_in              [[buffer(0)]],
    device      const T*    k_cache_pool_in         [[buffer(1)]],
    device      const T*    v_cache_pool_in         [[buffer(2)]],
    device      const uint* page_table_in           [[buffer(3)]],
    device      const int*  sequence_lengths_in     [[buffer(4)]],
    device      const int*  query_to_seq_map_in     [[buffer(5)]],
    device      const int*  query_token_offset_in   [[buffer(6)]],
    constant    const PagedAttentionParams& params  [[buffer(7)]],
    device      T*          output_buffer           [[buffer(8)]],
    uint        actual_simd_width                   [[threads_per_simdgroup]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]],
    uint        simd_lane_id                        [[thread_index_in_simdgroup]],
    uint        simd_group_id                       [[simdgroup_index_in_threadgroup]]
) {
    using Vec4 = typename Vec<T, 4>::Type;

    uint global_item_idx = tg_pos_in_grid.x;
    uint local_thread_idx = local_idx_in_tg;
    const uint num_simd_groups = max(1u, (tg_dim.x + actual_simd_width - 1) / actual_simd_width);
    const uint chunks_per_row = params.head_dim / 4;

    const uint q_head_for_kv_map     = (params.num_q_heads > 1)
                                     ? (global_item_idx % params.num_q_heads)
                                     : 0;
    uint target_kv_head_idx = q_head_for_kv_map;
    if (params.num_q_heads > params.num_kv_heads) {
        target_kv_head_idx = q_head_for_kv_map / (params.num_q_heads / params.num_kv_heads);
    }

    threadgroup float* q_shmem = tg_mem;

    uintptr_t current_offset = (uintptr_t)(q_shmem + params.head_dim);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_partial_reduce_scratch = (threadgroup float*)current_offset;

    current_offset = (uintptr_t)(tg_partial_reduce_scratch + tg_dim.x);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_simd_reduce_scratch = (threadgroup float*)current_offset;

    current_offset = (uintptr_t)(tg_simd_reduce_scratch + num_simd_groups);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_simd_exp_sums_scratch = (threadgroup float*)current_offset;

    current_offset = (uintptr_t)(tg_simd_exp_sums_scratch + num_simd_groups);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float2* tg_global_stats = (threadgroup float2*)current_offset;

    current_offset = (uintptr_t)(tg_global_stats + 1);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* tg_s_global_comp = (threadgroup float*)current_offset;

    current_offset = (uintptr_t)(tg_s_global_comp + 1);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float4* tg_simd_v_chunk_sums = (threadgroup float4*)current_offset;

    current_offset = (uintptr_t)(tg_simd_v_chunk_sums + num_simd_groups);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup T* K_tile = (threadgroup T*)current_offset;

    current_offset += params.tokens_per_page * params.head_dim * sizeof(T);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup T* V_tile = (threadgroup T*)current_offset;

    current_offset += params.tokens_per_page * params.head_dim * sizeof(T);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup uint* tg_page_table_slice = (threadgroup uint*)current_offset;

    current_offset += params.max_logical_blocks_per_seq * sizeof(uint);
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    current_offset += 32;
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;

    device const T* q_vector_item_ptr;
    if (params.num_q_heads > 1) {
        uint item_token_idx = global_item_idx / params.num_q_heads;
        uint item_q_head_idx = global_item_idx % params.num_q_heads;
        ulong query_base_offset = (ulong)item_token_idx * params.num_q_heads * params.head_dim +
                                 (ulong)item_q_head_idx * params.head_dim;
        q_vector_item_ptr = queries_in + query_base_offset;
    } else {
        q_vector_item_ptr = queries_in + (global_item_idx * params.head_dim);
    }

    threadgroup float4* q_vec_f4 = reinterpret_cast<threadgroup float4*>(q_shmem);
    for (uint c = local_thread_idx; c < chunks_per_row; c += tg_dim.x) {
        Vec4 h = reinterpret_cast<device const Vec4*>(q_vector_item_ptr)[c];
        q_vec_f4[c] = to_float4(h) * params.inv_sqrt_head_dim;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint token_idx_for_sideband_lookup;
    if (params.num_q_heads > 1) {
        token_idx_for_sideband_lookup = global_item_idx / params.num_q_heads;
    } else {
        token_idx_for_sideband_lookup = global_item_idx;
    }

    uint item_seq_idx_in_batch = (uint)query_to_seq_map_in[token_idx_for_sideband_lookup];
    for (uint blk = local_thread_idx; blk < params.max_logical_blocks_per_seq; blk += tg_dim.x) {
        uint flat_idx = item_seq_idx_in_batch * params.max_logical_blocks_per_seq + blk;
        tg_page_table_slice[blk] = page_table_in[flat_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int item_signed_query_token_offset = query_token_offset_in[token_idx_for_sideband_lookup];
    uint item_current_q_token_logical_pos = (uint)item_signed_query_token_offset;
    uint item_actual_sequence_length = (uint)sequence_lengths_in[item_seq_idx_in_batch];
    uint item_effective_history_length = min(item_current_q_token_logical_pos, item_actual_sequence_length);

    if (local_thread_idx == 0) {
        (*tg_global_stats).x = -INFINITY;
        (*tg_global_stats).y = 0.0f;
        (*tg_s_global_comp) = 0.0f;
    }

    const uint kMaxHeadDimMetal = 256;
    float acc_tile_local_fp32[kMaxHeadDimMetal];
    for (uint i_acc = 0; i_acc < params.head_dim; ++i_acc) {
        acc_tile_local_fp32[i_acc] = 0.0f;
    }

    const uint per_token_stride = params.num_kv_heads * params.head_dim;
    const uint per_page_stride = params.tokens_per_page * per_token_stride;
    const ulong kv_head_offset = (ulong)target_kv_head_idx * (ulong)params.head_dim;

    for (uint hist_tile_start = 0; hist_tile_start < item_effective_history_length; hist_tile_start += params.tokens_per_page) {
        uint current_hist_tile_actual_len = min(params.tokens_per_page, item_effective_history_length - hist_tile_start);
        const uint rows_in_tile_const = current_hist_tile_actual_len;

        for (uint row_idx_in_tile = simd_group_id; row_idx_in_tile < rows_in_tile_const; row_idx_in_tile += num_simd_groups) {
            uint absolute_hist_pos = hist_tile_start + row_idx_in_tile;
            uint logical_block_idx = absolute_hist_pos / params.tokens_per_page;
            uint token_slot_in_page = absolute_hist_pos % params.tokens_per_page;
            uint physical_page_id = tg_page_table_slice[logical_block_idx];
            ulong total_offset = (ulong)physical_page_id * (ulong)per_page_stride + (ulong)token_slot_in_page * (ulong)per_token_stride + kv_head_offset;

            device const T* k_vector_global_ptr = k_cache_pool_in + total_offset;
            device const T* v_vector_global_ptr = v_cache_pool_in + total_offset;

            threadgroup T* k_tile_row_base_ptr = K_tile + (row_idx_in_tile * params.head_dim);
            threadgroup T* v_tile_row_base_ptr = V_tile + (row_idx_in_tile * params.head_dim);
            threadgroup Vec4* k_dst_row_h4_ptr = reinterpret_cast<threadgroup Vec4*>(k_tile_row_base_ptr);
            threadgroup Vec4* v_dst_row_h4_ptr = reinterpret_cast<threadgroup Vec4*>(v_tile_row_base_ptr);

            for (uint chunk_idx_in_row = simd_lane_id; chunk_idx_in_row < chunks_per_row; chunk_idx_in_row += actual_simd_width) {
                device const Vec4* k_vec_h4_ptr = reinterpret_cast<device const Vec4*>(k_vector_global_ptr);
                device const Vec4* v_vec_h4_ptr = reinterpret_cast<device const Vec4*>(v_vector_global_ptr);
                Vec4 k_h4_val = k_vec_h4_ptr[chunk_idx_in_row];
                Vec4 v_h4_val = v_vec_h4_ptr[chunk_idx_in_row];
                k_dst_row_h4_ptr[chunk_idx_in_row] = k_h4_val;
                v_dst_row_h4_ptr[chunk_idx_in_row] = v_h4_val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint rows_per_lane  = (current_hist_tile_actual_len + actual_simd_width - 1) / actual_simd_width;
        const uint kMaxLocalRows  = 8;
        float      local_scores[kMaxLocalRows];
        uint       local_row_ids[kMaxLocalRows];
        uint       local_rows_filled = 0;
        float thread_score_val = -INFINITY;

        for (uint r = 0; r < rows_per_lane; ++r) {
            uint row_idx = simd_lane_id + r * actual_simd_width;
            if (row_idx >= current_hist_tile_actual_len) break;

            threadgroup const T* k_vector_from_tile_h = K_tile + (row_idx * params.head_dim);
            float4 dot_acc = float4(0.0f);
            for (uint d = 0; d < params.head_dim; d += 4) {
                float4 qf = *((threadgroup const float4*)(q_shmem + d));
                Vec4 k_vec = *((threadgroup const Vec4*)(k_vector_from_tile_h + d));
                float4 kf = to_float4(k_vec);
                dot_acc = fma(qf, kf, dot_acc);
            }
            float row_score = dot_acc.x + dot_acc.y + dot_acc.z + dot_acc.w;
            thread_score_val = max(thread_score_val, row_score);

            if (local_rows_filled < kMaxLocalRows) {
                local_scores[local_rows_filled] = row_score;
                local_row_ids[local_rows_filled] = row_idx;
                ++local_rows_filled;
            }
        }

        float m_local_tile_val = warp_max(thread_score_val, actual_simd_width);
        if (simd_lane_id == 0) tg_partial_reduce_scratch[simd_group_id] = m_local_tile_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
            float v = (local_thread_idx < num_simd_groups) ? tg_partial_reduce_scratch[local_thread_idx] : -INFINITY;
            m_local_tile_val = warp_max(v, actual_simd_width);
            if (local_thread_idx == 0) tg_simd_reduce_scratch[0] = m_local_tile_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        m_local_tile_val = tg_simd_reduce_scratch[0];

        float thread_exp_total = 0.0f;
        for (uint i_row = 0; i_row < local_rows_filled; ++i_row) {
            thread_exp_total += fast::exp(max(local_scores[i_row] - m_local_tile_val, params.log_exp_min_clamp));
        }

        float d_local_tile_total_val = warp_sum(thread_exp_total, actual_simd_width);
        if (simd_lane_id == 0) tg_simd_exp_sums_scratch[simd_group_id] = d_local_tile_total_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group_id == 0) {
            float v = (local_thread_idx < num_simd_groups) ? tg_simd_exp_sums_scratch[local_thread_idx] : 0.0f;
            d_local_tile_total_val = warp_sum(v, actual_simd_width);
            if (local_thread_idx == 0) tg_simd_exp_sums_scratch[0] = d_local_tile_total_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        d_local_tile_total_val = tg_simd_exp_sums_scratch[0];

        if (local_thread_idx == 0) {
            float m_prev = (*tg_global_stats).x;
            float s_prev = (*tg_global_stats).y;
            float c_s_prev = *tg_s_global_comp;
            float m_new = m_prev;
            float s_new_uncompensated = s_prev;
            float c_s_new = c_s_prev;
            float scale_f = 1.0f;

            if (m_local_tile_val > m_prev) {
                m_new = m_local_tile_val;
                scale_f = fast::exp(max(m_prev - m_new, params.log_exp_min_clamp));
                s_new_uncompensated = s_prev * scale_f;
                c_s_new = c_s_prev * scale_f;
            }

            float exp_arg = max(m_local_tile_val - m_new, params.log_exp_min_clamp);
            float y_kahan = (d_local_tile_total_val * fast::exp(exp_arg)) - c_s_new;
            float t_kahan = s_new_uncompensated + y_kahan;
            c_s_new = (t_kahan - s_new_uncompensated) - y_kahan;
            float s_new_final = t_kahan;

            (*tg_global_stats).x = m_new;
            (*tg_global_stats).y = s_new_final;
            *tg_s_global_comp = c_s_new;
            tg_simd_reduce_scratch[0] = scale_f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_global_current_iter_atomic = (*tg_global_stats).x;
        float scale_for_acc_iter_atomic = tg_simd_reduce_scratch[0];

        if (scale_for_acc_iter_atomic != 1.0f) {
            for (uint d = 0; d < params.head_dim; d += 4) {
                float4 acc_chunk = float4(acc_tile_local_fp32[d],
                                        (d + 1 < params.head_dim) ? acc_tile_local_fp32[d+1] : 0.0f,
                                        (d + 2 < params.head_dim) ? acc_tile_local_fp32[d+2] : 0.0f,
                                        (d + 3 < params.head_dim) ? acc_tile_local_fp32[d+3] : 0.0f);
                acc_chunk *= scale_for_acc_iter_atomic;
                acc_tile_local_fp32[d] = acc_chunk.x;
                if (d + 1 < params.head_dim) acc_tile_local_fp32[d+1] = acc_chunk.y;
                if (d + 2 < params.head_dim) acc_tile_local_fp32[d+2] = acc_chunk.z;
                if (d + 3 < params.head_dim) acc_tile_local_fp32[d+3] = acc_chunk.w;
            }
        }

        for (uint i_row = 0; i_row < local_rows_filled; ++i_row) {
            uint row_idx = local_row_ids[i_row];
            threadgroup const T* v_vector_from_tile_h = V_tile + (row_idx * params.head_dim);

            float weight_term = fast::exp(max(local_scores[i_row] - m_local_tile_val, params.log_exp_min_clamp));
            float exp_term    = fast::exp(max(m_local_tile_val - m_global_current_iter_atomic, params.log_exp_min_clamp));
            float final_p_attn_weight_numerator = weight_term * exp_term;

            threadgroup const Vec4* v_vec_h4_ptr = reinterpret_cast<threadgroup const Vec4*>(v_vector_from_tile_h);
            thread float4* acc_f4_ptr = reinterpret_cast<thread float4*>(acc_tile_local_fp32);

            for (uint chunk_idx = 0; chunk_idx < chunks_per_row; ++chunk_idx) {
                float4 v_chunk_fp32 = to_float4(v_vec_h4_ptr[chunk_idx]) * final_p_attn_weight_numerator;
                acc_f4_ptr[chunk_idx] += v_chunk_fp32;
            }
        }
    }

    float s_global_final = (*tg_global_stats).y;
    float inv_s_global = (s_global_final > kEpsilonForZeroGuard) ? fast::divide(1.0f, s_global_final) : 0.0f;

    for (uint i = 0; i < params.head_dim; i += 4) {
        thread float4* chunk_ptr = reinterpret_cast<thread float4*>(acc_tile_local_fp32 + i);
        float4 chunk = *chunk_ptr;
        chunk *= inv_s_global;
        *chunk_ptr = chunk;
    }

    for (uint i = 0; i < params.head_dim; i += 4) {
        float4 chunk_to_write = float4(0.0f);
        if (i < params.head_dim)     chunk_to_write.x = acc_tile_local_fp32[i+0];
        if (i+1 < params.head_dim) chunk_to_write.y = acc_tile_local_fp32[i+1];
        if (i+2 < params.head_dim) chunk_to_write.z = acc_tile_local_fp32[i+2];
        if (i+3 < params.head_dim) chunk_to_write.w = acc_tile_local_fp32[i+3];

        float4 reduced_simd_group_final_chunk;
        reduced_simd_group_final_chunk.x = warp_sum(chunk_to_write.x, actual_simd_width);
        reduced_simd_group_final_chunk.y = warp_sum(chunk_to_write.y, actual_simd_width);
        reduced_simd_group_final_chunk.z = warp_sum(chunk_to_write.z, actual_simd_width);
        reduced_simd_group_final_chunk.w = warp_sum(chunk_to_write.w, actual_simd_width);

        if (simd_lane_id == 0) {
            tg_simd_v_chunk_sums[simd_group_id] = reduced_simd_group_final_chunk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (local_thread_idx == 0) {
            float4 final_output_chunk = float4(0.0f);
            for (uint sg_idx = 0; sg_idx < num_simd_groups; ++sg_idx) {
                final_output_chunk += tg_simd_v_chunk_sums[sg_idx];
            }

            uint output_base_idx = global_item_idx * params.head_dim + i;
            output_buffer[output_base_idx + 0] = (T)final_output_chunk.x;
            output_buffer[output_base_idx + 1] = (T)final_output_chunk.y;
            output_buffer[output_base_idx + 2] = (T)final_output_chunk.z;
            output_buffer[output_base_idx + 3] = (T)final_output_chunk.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
