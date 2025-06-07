// paged_attention_2pass.h.metal
// Metal shader implementation for paged attention operations with tiled V accumulation.
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

template <typename T>
[[kernel]] void paged_attn_pass1_kernel(
    device      const T*        queries_in                      [[buffer(0)]],
    device      const T*        k_cache_pool_in                 [[buffer(1)]],
    device      const T*        v_cache_pool_in                 [[buffer(2)]],
    device      const uint*     page_table_in                   [[buffer(3)]],
    device      const int*      sequence_lengths_in             [[buffer(4)]],
    device      const int*      query_token_offset_in           [[buffer(5)]],
    constant    const           PagedAttentionParams& params    [[buffer(6)]],
    device      const uint2*    active_work_item_pairs          [[buffer(7)]],
    device      const uint*     query_starts_for_batch_item_arr [[buffer(8)]],
    device      float*          m_locals_pass1_out              [[buffer(9)]],
    device      float*          s_locals_pass1_out              [[buffer(10)]],
    device      T*              o_partials_pass1_out            [[buffer(11)]],
    uint        actual_simd_width                               [[threads_per_simdgroup]],
    threadgroup float* tg_mem                                   [[threadgroup(0)]],
    uint3       tg_pos_in_grid                                  [[threadgroup_position_in_grid]],
    uint3       tg_dim                                          [[threads_per_threadgroup]],
    uint        local_idx_in_tg                                 [[thread_index_in_threadgroup]],
    uint        simd_lane_id                                    [[thread_index_in_simdgroup]],
    uint        simd_group_id                                   [[simdgroup_index_in_threadgroup]]
)
{
    using Vec4 = typename Vec<T, 4>::Type;

    threadgroup uchar* tg_mem_base_byte_ptr = (threadgroup uchar*)tg_mem;
    uintptr_t current_offset = 0;

    const uint D_s = params.tokens_per_page;
    const uint N_q_per_kv = (params.num_q_heads + params.num_kv_heads - 1) / params.num_kv_heads;

    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* Q_shmem_base = (threadgroup float*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += N_q_per_kv * D_s * params.head_dim * sizeof(float);

    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup T* K_tile = (threadgroup T*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += D_s * params.head_dim * sizeof(T);

    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup T* V_tile = (threadgroup T*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += D_s * params.head_dim * sizeof(T);

    const uint total_simd_groups_in_tg_metal = tg_dim.x / actual_simd_width;
    current_offset = (current_offset + kAlignmentMask) & ~kAlignmentMask;
    threadgroup float* V_Sum_Accumulators_Area = (threadgroup float*)(tg_mem_base_byte_ptr + current_offset);
    current_offset += total_simd_groups_in_tg_metal * params.head_dim * sizeof(float);

    uint flat_work_item_idx = tg_pos_in_grid.x;
    if (flat_work_item_idx >= params.num_active_batch_logical_pages) {
        return;
    }
    uint2 work_item_pair = active_work_item_pairs[flat_work_item_idx];
    uint assigned_batch_item_idx = work_item_pair.x;
    uint assigned_logical_page_idx_in_sequence = work_item_pair.y;
    uint assigned_global_kv_head_idx = tg_pos_in_grid.y;

    uint page_table_flat_idx = assigned_batch_item_idx * params.max_logical_blocks_per_seq + assigned_logical_page_idx_in_sequence;
    uint physical_page_id = page_table_in[page_table_flat_idx];
    if (physical_page_id >= params.num_physical_pages_in_pool) {
        return;
    }

    const uint threads_per_tg = tg_dim.x;
    const uint chunked_head_dim_size = params.head_dim / 4;

    const ulong page_base_offset_global_kv = (ulong)physical_page_id *
                                            (ulong)params.tokens_per_page *
                                            (ulong)params.num_kv_heads *
                                            (ulong)params.head_dim;
    for (uint token_idx_on_page = 0; token_idx_on_page < D_s; ++token_idx_on_page) {
        ulong k_global_offset = page_base_offset_global_kv +
                                (ulong)token_idx_on_page * (ulong)params.num_kv_heads * (ulong)params.head_dim +
                                (ulong)assigned_global_kv_head_idx * (ulong)params.head_dim;
        device const T* k_src_ptr = k_cache_pool_in + k_global_offset;
        threadgroup T* k_dst_ptr = K_tile + token_idx_on_page * params.head_dim;
        threadgroup Vec4* dst_k_vector_h4_ptr = (threadgroup Vec4*)(k_dst_ptr);
        device const Vec4* src_k_vector_h4_ptr = (device const Vec4*)(k_src_ptr);
        for (uint chunk_idx = local_idx_in_tg; chunk_idx < chunked_head_dim_size; chunk_idx += threads_per_tg) {
            dst_k_vector_h4_ptr[chunk_idx] = src_k_vector_h4_ptr[chunk_idx];
        }
    }

    for (uint token_idx_on_page = 0; token_idx_on_page < D_s; ++token_idx_on_page) {
        ulong v_global_offset = page_base_offset_global_kv +
                                (ulong)token_idx_on_page * (ulong)params.num_kv_heads * (ulong)params.head_dim +
                                (ulong)assigned_global_kv_head_idx * (ulong)params.head_dim;
        device const T* v_src_ptr = v_cache_pool_in + v_global_offset;
        threadgroup T* v_dst_ptr = V_tile + token_idx_on_page * params.head_dim;
        threadgroup Vec4* dst_v_vector_h4_ptr = (threadgroup Vec4*)(v_dst_ptr);
        device const Vec4* src_v_vector_h4_ptr = (device const Vec4*)(v_src_ptr);
        for (uint chunk_idx = local_idx_in_tg; chunk_idx < chunked_head_dim_size; chunk_idx += threads_per_tg) {
            dst_v_vector_h4_ptr[chunk_idx] = src_v_vector_h4_ptr[chunk_idx];
        }
    }

    const uint simd_groups_per_gqa_stream = total_simd_groups_in_tg_metal / N_q_per_kv;
    const uint gqa_stream_idx_for_this_simd_group = simd_group_id / simd_groups_per_gqa_stream;
    const uint sub_simd_group_idx_within_stream = simd_group_id % simd_groups_per_gqa_stream;

    uint seq_len_for_this_batch_item = (uint)sequence_lengths_in[assigned_batch_item_idx];
    for (uint q_block_start_local_idx = 0;
        q_block_start_local_idx < seq_len_for_this_batch_item;
        q_block_start_local_idx += D_s) {

        uint num_queries_in_this_block = min(D_s, seq_len_for_this_batch_item - q_block_start_local_idx);
        if (gqa_stream_idx_for_this_simd_group < N_q_per_kv) {
            uint target_q_head_local_offset_in_gqa_group = gqa_stream_idx_for_this_simd_group;
            uint target_global_q_head_idx = (assigned_global_kv_head_idx * N_q_per_kv) + target_q_head_local_offset_in_gqa_group;

            if (target_global_q_head_idx >= params.num_q_heads) {
                continue;
            }

            threadgroup float* q_block_shmem_for_gqa_stream = Q_shmem_base +
                                    (gqa_stream_idx_for_this_simd_group * D_s * params.head_dim);

            for (uint q_idx_in_block_for_this_sg = sub_simd_group_idx_within_stream;
                q_idx_in_block_for_this_sg < num_queries_in_this_block;
                q_idx_in_block_for_this_sg += simd_groups_per_gqa_stream) {

                uint current_query_local_idx = q_block_start_local_idx + q_idx_in_block_for_this_sg;
                uint master_query_idx = query_starts_for_batch_item_arr[assigned_batch_item_idx] + current_query_local_idx;

                device const T* q_src_global_ptr = queries_in +
                    (master_query_idx * params.num_q_heads * params.head_dim) +
                    (target_global_q_head_idx * params.head_dim);

                threadgroup float* q_dest_specific_q_in_shmem = q_block_shmem_for_gqa_stream +
                                                                (q_idx_in_block_for_this_sg * params.head_dim);

                for (uint c_idx = simd_lane_id; c_idx < chunked_head_dim_size; c_idx += actual_simd_width) {
                    ((threadgroup float4*)q_dest_specific_q_in_shmem)[c_idx] =
                        to_float4(((device const Vec4*)q_src_global_ptr)[c_idx]) * params.inv_sqrt_head_dim;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint total_floats_in_v_acc_area = total_simd_groups_in_tg_metal * params.head_dim;
        uint num_float4s_to_zero = total_floats_in_v_acc_area / 4;

        threadgroup float4* v_acc_area_f4_ptr = (threadgroup float4*)V_Sum_Accumulators_Area;
        for (uint i = local_idx_in_tg; i < num_float4s_to_zero; i += tg_dim.x) {
            v_acc_area_f4_ptr[i] = float4(0.0f);
        }

        if (gqa_stream_idx_for_this_simd_group >= N_q_per_kv) {
            continue;
        }

        threadgroup float* q_block_shmem_for_this_gqa_stream_for_compute = Q_shmem_base +
                                    (gqa_stream_idx_for_this_simd_group * D_s * params.head_dim);

        for (uint q_idx_in_block_for_this_sg = sub_simd_group_idx_within_stream;
              q_idx_in_block_for_this_sg < num_queries_in_this_block;
              q_idx_in_block_for_this_sg += simd_groups_per_gqa_stream
        ) {
            uint current_query_local_idx_in_sequence = q_block_start_local_idx + q_idx_in_block_for_this_sg;
            uint master_query_idx = query_starts_for_batch_item_arr[assigned_batch_item_idx] + current_query_local_idx_in_sequence;
            uint current_q_logical_pos = (uint)query_token_offset_in[master_query_idx];
            uint target_global_q_head_idx = (assigned_global_kv_head_idx * N_q_per_kv) + gqa_stream_idx_for_this_simd_group;

            if (target_global_q_head_idx >= params.num_q_heads) {
                continue;
            }

            threadgroup const float* q_vec_ptr = q_block_shmem_for_this_gqa_stream_for_compute +
                                                 (q_idx_in_block_for_this_sg * params.head_dim);
            threadgroup float* v_sum_accumulator_ptr = V_Sum_Accumulators_Area +
                                                       (simd_group_id * params.head_dim);

            threadgroup float4* v_acc_f4_ptr = (threadgroup float4*)v_sum_accumulator_ptr;
            for (uint f4_idx = simd_lane_id; f4_idx < chunked_head_dim_size; f4_idx += actual_simd_width) {
                v_acc_f4_ptr[f4_idx] = float4(0.0f);
            }

            float page_max_score = -INFINITY;
            float page_sum_exp_norm_by_page_max = 0.0f;
            float kahan_c_for_sum_exp = 0.0f;

            for (uint k_idx_in_tile = 0; k_idx_in_tile < D_s; ++k_idx_in_tile) {
                threadgroup const T* k_vec_hist_ptr = K_tile + (k_idx_in_tile * params.head_dim);
                uint history_token_logical_pos = (assigned_logical_page_idx_in_sequence * D_s) + k_idx_in_tile;

                if (history_token_logical_pos > current_q_logical_pos ||
                    history_token_logical_pos >= seq_len_for_this_batch_item) {
                    continue;
                }

                float per_lane_partial_score = 0.0f;
                for (uint f4_chunk_idx = simd_lane_id; f4_chunk_idx < chunked_head_dim_size; f4_chunk_idx += actual_simd_width) {
                    uint d_offset = f4_chunk_idx * 4;
                    float4 qv = *((threadgroup const float4*)(q_vec_ptr + d_offset));
                    float4 kv = to_float4(*((threadgroup const Vec4*)(k_vec_hist_ptr + d_offset)));
                    per_lane_partial_score += dot(qv, kv);
                }
                float score = simd_sum(per_lane_partial_score);
                score = simd_broadcast_first(score);

                float old_page_max_score_val = page_max_score;
                page_max_score = max(page_max_score, score);

                float current_score_exp_contribution;

                if (page_max_score > old_page_max_score_val && old_page_max_score_val != -INFINITY) {
                    float rescale_exp_arg = max(old_page_max_score_val - page_max_score, params.log_exp_min_clamp);
                    float actual_scale_factor = fast::exp(rescale_exp_arg);

                    page_sum_exp_norm_by_page_max *= actual_scale_factor;
                    kahan_c_for_sum_exp *= actual_scale_factor;

                    for (uint h_rescale_idx = simd_lane_id;
                        h_rescale_idx < chunked_head_dim_size;
                        h_rescale_idx += actual_simd_width)
                    {
                        uint d = h_rescale_idx * 4;
                        *((threadgroup float4*)(v_sum_accumulator_ptr + d)) *= actual_scale_factor;
                    }
                }

                float current_term_exp_arg = max(score - page_max_score, params.log_exp_min_clamp);
                current_score_exp_contribution = fast::exp(current_term_exp_arg);

                float y_kahan = current_score_exp_contribution - kahan_c_for_sum_exp;
                float t_kahan = page_sum_exp_norm_by_page_max + y_kahan;
                kahan_c_for_sum_exp = (t_kahan - page_sum_exp_norm_by_page_max) - y_kahan;
                page_sum_exp_norm_by_page_max = t_kahan;

                threadgroup const T* v_vec_hist_ptr = V_tile + (k_idx_in_tile * params.head_dim);
                for (uint h_chunk_idx = simd_lane_id;
                      h_chunk_idx < chunked_head_dim_size;
                      h_chunk_idx += actual_simd_width) {
                    uint h_dim_offset = h_chunk_idx * 4;
                    float4 v_chunk_f = to_float4( *((threadgroup const Vec4*)(v_vec_hist_ptr + h_dim_offset)) );
                    float4 current_acc = *((threadgroup float4*)(v_sum_accumulator_ptr + h_dim_offset));

                    float4 updated_acc;
                    updated_acc.x = fma(current_score_exp_contribution, v_chunk_f.x, current_acc.x);
                    updated_acc.y = fma(current_score_exp_contribution, v_chunk_f.y, current_acc.y);
                    updated_acc.z = fma(current_score_exp_contribution, v_chunk_f.z, current_acc.z);
                    updated_acc.w = fma(current_score_exp_contribution, v_chunk_f.w, current_acc.w);

                    *((threadgroup float4*)(v_sum_accumulator_ptr + h_dim_offset)) = updated_acc;
                }
            }

             if (simd_lane_id == 0) {
                ulong ms_base_offset = (ulong)master_query_idx * params.num_q_heads * params.num_active_batch_logical_pages +
                                       (ulong)target_global_q_head_idx * params.num_active_batch_logical_pages;
                ulong ms_flat_output_idx = ms_base_offset + flat_work_item_idx;

                m_locals_pass1_out[ms_flat_output_idx] = page_max_score;
                s_locals_pass1_out[ms_flat_output_idx] = page_sum_exp_norm_by_page_max;
            }

            ulong o_base_offset = (ulong)master_query_idx * params.num_q_heads * params.num_active_batch_logical_pages * params.head_dim +
                                    (ulong)target_global_q_head_idx * params.num_active_batch_logical_pages * params.head_dim +
                                    (ulong)flat_work_item_idx * params.head_dim;

            device T* o_dest_ptr = o_partials_pass1_out + o_base_offset;

            for (uint h_chunk_idx = simd_lane_id;
                h_chunk_idx < chunked_head_dim_size;
                h_chunk_idx += actual_simd_width) {

                uint h_dim_offset = h_chunk_idx * 4;
                float4 val_f4 = *((threadgroup float4*)(v_sum_accumulator_ptr + h_dim_offset));
                *((device Vec4*)(o_dest_ptr + h_dim_offset)) = from_float4<T>(val_f4);
            }
        }
    }
}

template <typename T>
[[kernel]] void paged_attn_pass2_kernel(
    device      const float* m_pass1_results        [[buffer(0)]],
    device      const float* s_pass1_results        [[buffer(1)]],
    device      const T*     o_pass1_results        [[buffer(2)]],
    constant    const PagedAttentionParams& params  [[buffer(3)]],
    device      T* final_output_buffer              [[buffer(4)]],
    uint        actual_simd_width                   [[threads_per_simdgroup]],
    threadgroup float* tg_mem                       [[threadgroup(0)]],
    uint3       tg_pos_in_grid                      [[threadgroup_position_in_grid]],
    uint3       tg_dim                              [[threads_per_threadgroup]],
    uint        local_idx_in_tg                     [[thread_index_in_threadgroup]]
) {
    using Vec4 = typename Vec<T, 4>::Type;
    const uint NumSIMDgroups_Pass2 = tg_dim.x / actual_simd_width;
    const uint simd_group_id = local_idx_in_tg / actual_simd_width;
    const uint simd_lane_id = local_idx_in_tg % actual_simd_width;

    threadgroup uchar* tg_mem_base_byte_ptr = (threadgroup uchar*)tg_mem;
    uintptr_t current_tg_offset = 0;

    auto align_offset = [] (uintptr_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    };
    const size_t float_alignment = alignof(float);
    const size_t float4_alignment = alignof(float4);

    current_tg_offset = align_offset(current_tg_offset, float_alignment);
    threadgroup float* M_item_shared_scalar = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += sizeof(float);

    current_tg_offset = align_offset(current_tg_offset, float_alignment);
    threadgroup float* S_item_shared_scalar = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += sizeof(float);

    current_tg_offset = align_offset(current_tg_offset, float_alignment);
    threadgroup float* S_item_kahan_c_shared = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += sizeof(float);

    current_tg_offset = align_offset(current_tg_offset, float4_alignment);
    threadgroup float* O_item_shared_accumulator = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += params.head_dim * sizeof(float);

    current_tg_offset = align_offset(current_tg_offset, float_alignment);
    threadgroup float* simdgroup_m_scratch = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += NumSIMDgroups_Pass2 * sizeof(float);

    current_tg_offset = align_offset(current_tg_offset, float_alignment);
    threadgroup float* simdgroup_s_scratch = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);
    current_tg_offset += NumSIMDgroups_Pass2 * sizeof(float);

    current_tg_offset = align_offset(current_tg_offset, float4_alignment);
    threadgroup float* simdgroup_o_partials = (threadgroup float*)(tg_mem_base_byte_ptr + current_tg_offset);

    const uint items_in_token_dim = params.pass2_token_block_size;
    const uint items_in_qhead_dim = params.pass2_qhead_block_size;

    const uint items_in_flat_dim = items_in_token_dim * items_in_qhead_dim;
    const uint chunked_head_dim_size = params.head_dim / 4;

    for (uint item_flat_idx_in_block = 0; item_flat_idx_in_block < items_in_flat_dim; ++item_flat_idx_in_block) {
        uint local_token_idx_in_block = item_flat_idx_in_block % items_in_token_dim;
        uint local_q_head_idx_in_block = item_flat_idx_in_block / items_in_token_dim;

        uint current_master_query_idx = (tg_pos_in_grid.x * items_in_token_dim) + local_token_idx_in_block;
        uint current_target_q_head_idx = (tg_pos_in_grid.y * items_in_qhead_dim) + local_q_head_idx_in_block;

        if (current_master_query_idx >= params.query_token_count_total ||
            current_target_q_head_idx >= params.num_q_heads) {
            continue;
        }

        if (local_idx_in_tg == 0) {
            *M_item_shared_scalar = -INFINITY;
            *S_item_shared_scalar = 0.0f;
            *S_item_kahan_c_shared = 0.0f;
        }

        for (uint i = local_idx_in_tg; i < params.head_dim; i += tg_dim.x) {
            O_item_shared_accumulator[i] = 0.0f;
        }

        for (uint i = local_idx_in_tg; i < NumSIMDgroups_Pass2; i += tg_dim.x) {
            simdgroup_m_scratch[i] = -INFINITY;
        }

        for (uint i = local_idx_in_tg; i < NumSIMDgroups_Pass2; i += tg_dim.x) {
            simdgroup_s_scratch[i] = 0.0f;
        }

        for (uint i = local_idx_in_tg; i < (NumSIMDgroups_Pass2 * params.head_dim); i += tg_dim.x) {
            simdgroup_o_partials[i] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        float m_thread_private_max = -INFINITY;
        for (uint page_idx = local_idx_in_tg; page_idx < params.num_active_batch_logical_pages; page_idx += tg_dim.x) {
            ulong m_s_stride_qhead_dim = params.num_active_batch_logical_pages;
            ulong m_s_stride_query_dim = params.num_q_heads * m_s_stride_qhead_dim;
            ulong flat_idx_m_value = (ulong)current_master_query_idx * m_s_stride_query_dim +
                                     (ulong)current_target_q_head_idx * m_s_stride_qhead_dim +
                                     page_idx;

            float m_value_from_page = m_pass1_results[flat_idx_m_value];
            m_thread_private_max = max(m_thread_private_max, m_value_from_page);
        }

        float m_simdgroup_max = simd_max(m_thread_private_max);

        if (simd_lane_id == 0) {
            simdgroup_m_scratch[simd_group_id] = m_simdgroup_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group_id == 0) {
            float final_max_for_item_local_to_sg0 = -INFINITY;
            for (uint i = simd_lane_id; i < NumSIMDgroups_Pass2; i += actual_simd_width) {
                final_max_for_item_local_to_sg0 = max(final_max_for_item_local_to_sg0, simdgroup_m_scratch[i]);
            }
            float final_max_for_item_reduced_in_sg0 = simd_max(final_max_for_item_local_to_sg0);

            if (simd_lane_id == 0) {
                *M_item_shared_scalar = final_max_for_item_reduced_in_sg0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float M_final_item_val = *M_item_shared_scalar;
        float s_sg_private_sum = 0.0f;

        threadgroup float* my_sg_o_partial_accumulator = simdgroup_o_partials +
                                                         (simd_group_id * params.head_dim);

        uint pages_per_sg_base = params.num_active_batch_logical_pages / NumSIMDgroups_Pass2;
        uint pages_remainder_sg = params.num_active_batch_logical_pages % NumSIMDgroups_Pass2;
        uint start_page_for_this_sg = simd_group_id * pages_per_sg_base + min(simd_group_id, pages_remainder_sg);
        uint end_page_for_this_sg = start_page_for_this_sg + pages_per_sg_base + (simd_group_id < pages_remainder_sg ? 1 : 0);

        for (uint page_idx = start_page_for_this_sg; page_idx < end_page_for_this_sg; ++page_idx) {
            float m_local_p_this_page = 0.0f;
            float s_local_p_this_page = 0.0f;
            float rescale_factor_this_page = 0.0f;

            if (simd_lane_id == 0) {
                ulong m_s_stride_qhead_dim = params.num_active_batch_logical_pages;
                ulong m_s_stride_query_dim = params.num_q_heads * m_s_stride_qhead_dim;
                ulong flat_idx_ms_value = (ulong)current_master_query_idx * m_s_stride_query_dim +
                                        (ulong)current_target_q_head_idx * m_s_stride_qhead_dim +
                                        page_idx;

                m_local_p_this_page = m_pass1_results[flat_idx_ms_value];
                s_local_p_this_page = s_pass1_results[flat_idx_ms_value];

                rescale_factor_this_page = fast::exp(max(m_local_p_this_page - M_final_item_val,
                                                            params.log_exp_min_clamp));
            }

            s_local_p_this_page = simd_broadcast_first(s_local_p_this_page);
            rescale_factor_this_page = simd_broadcast_first(rescale_factor_this_page);

            if (simd_lane_id == 0) {
                s_sg_private_sum += s_local_p_this_page * rescale_factor_this_page;
            }

            if (rescale_factor_this_page >= kEpsilonForZeroGuard) {
                ulong o_stride_page_dim = params.head_dim;
                ulong o_stride_qhead_dim = params.num_active_batch_logical_pages * o_stride_page_dim;
                ulong o_stride_query_dim = params.num_q_heads * o_stride_qhead_dim;
                ulong base_offset_o_global = (ulong)current_master_query_idx * o_stride_query_dim +
                                            (ulong)current_target_q_head_idx * o_stride_qhead_dim +
                                            (ulong)page_idx * o_stride_page_dim;
                device const T* o_partial_p_global_ptr = o_pass1_results + base_offset_o_global;

                for (uint h_offset_in_head = simd_lane_id;
                    h_offset_in_head < params.head_dim;
                    h_offset_in_head += actual_simd_width) {
                    float o_val_from_page_float = (float)o_partial_p_global_ptr[h_offset_in_head];
                    my_sg_o_partial_accumulator[h_offset_in_head] =
                        fma(o_val_from_page_float,
                            rescale_factor_this_page,
                            my_sg_o_partial_accumulator[h_offset_in_head]);
                }
            }
        }
        if (simd_lane_id == 0) {
            simdgroup_s_scratch[simd_group_id] = s_sg_private_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((simd_group_id == 0) && (simd_lane_id == 0)) {
            for (uint i = 0; i < NumSIMDgroups_Pass2; ++i) {
                float s_contrib_from_sg_scratch = simdgroup_s_scratch[i];

                float y_kahan = s_contrib_from_sg_scratch - (*S_item_kahan_c_shared);
                float t_kahan = (*S_item_shared_scalar) + y_kahan;
                *S_item_kahan_c_shared = (t_kahan - (*S_item_shared_scalar)) - y_kahan;
                *S_item_shared_scalar = t_kahan;
            }
        }

        for (uint h_target = local_idx_in_tg;
              h_target < params.head_dim;
              h_target += tg_dim.x) {
            float sum_for_this_h_component = 0.0f;
            for (uint sg_idx = 0; sg_idx < NumSIMDgroups_Pass2; ++sg_idx) {
                sum_for_this_h_component += simdgroup_o_partials[sg_idx * params.head_dim + h_target];
            }
            O_item_shared_accumulator[h_target] = sum_for_this_h_component;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float S_final_val = *S_item_shared_scalar;
        float inv_S_final = fast::divide(1.0f, S_final_val);

        ulong output_item_flat_idx = (ulong)current_master_query_idx * params.num_q_heads + current_target_q_head_idx;
        ulong base_offset_global_output = output_item_flat_idx * params.head_dim;
        device T* final_out_ptr_for_item_base = final_output_buffer + base_offset_global_output;

        for (uint vec_idx = local_idx_in_tg; vec_idx < chunked_head_dim_size; vec_idx += tg_dim.x) {
            uint h_start_idx = vec_idx * 4;

            float4 o_chunk_float = float4(O_item_shared_accumulator[h_start_idx + 0],
                                          O_item_shared_accumulator[h_start_idx + 1],
                                          O_item_shared_accumulator[h_start_idx + 2],
                                          O_item_shared_accumulator[h_start_idx + 3]);

            o_chunk_float *= inv_S_final;

            Vec4 o_chunk_typed = from_float4<T>(o_chunk_float);

            device Vec4* dest_ptr_typed = (device Vec4*)(final_out_ptr_for_item_base + h_start_idx);
            *dest_ptr_typed = o_chunk_typed;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
