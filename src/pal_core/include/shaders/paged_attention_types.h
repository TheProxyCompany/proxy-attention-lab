// paged_attention_types.h
// Defines parameter structures shared between CPU and GPU for paged attention.
//
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
// ============================================================================

#pragma once

#ifdef __METAL_VERSION__
// Metal shader - use Metal Standard Library types
using uint32_t = unsigned int;
using float32_t = float;
#else
// C++ code - use standard headers
#include <stdint.h>
#include <type_traits>
#endif


/**
 * @brief Shared parameter structure for paged attention operations.
 *
 * This structure is shared between CPU (C++) and GPU (Metal) code to ensure
 * consistent parameter passing. The structure is explicitly aligned to 16 bytes
 * to match Metal's buffer requirements for optimal performance.
 */
struct alignas(16) PagedAttentionParams {
  uint32_t num_q_heads;                   // Number of query heads
  uint32_t num_kv_heads;                  // Number of key/value heads
  uint32_t head_dim;                      // Hidden dimension per head
  uint32_t tokens_per_page;               // Number of tokens stored in each page
  uint32_t max_logical_blocks_per_seq;    // Maximum logical blocks per sequence
  uint32_t num_physical_pages_in_pool;    // Number of physical pages in pool
  uint32_t num_sequences_in_batch;        // Number of sequences in batch
  uint32_t num_active_batch_logical_pages; // Number of active (batch_item, logical_page) pairs
  uint32_t pass2_token_block_size;        // Token block size for Pass 2 2D dispatch
  uint32_t pass2_qhead_block_size;        // Q-head block size for Pass 2 2D dispatch
  uint32_t query_token_count_total;       // Total number of query tokens in batch
  float    log_exp_min_clamp;             // Minimum value for exponent in exp function
  float    inv_sqrt_head_dim;             // 1/sqrt(head_dim) precomputed on host
};

// --- Assertions ---
#ifndef __METAL_VERSION__ // C++ side

static_assert(std::is_standard_layout_v<PagedAttentionParams>,
              "PagedAttentionParams must be a standard-layout type.");
static_assert(alignof(PagedAttentionParams) == 16,
              "PagedAttentionParams must have 16-byte alignment.");
// 11 uint32_t (44 bytes) + 2 float (8 bytes) = 52 data bytes.
// alignas(16) means total size is 64 bytes (rounded up to multiple of 16).
static_assert(sizeof(PagedAttentionParams) == 64, "C++ sizeof(PagedAttentionParams) expected to be 64 bytes.");

#else // __METAL_VERSION__ (Metal side)
static_assert(sizeof(PagedAttentionParams) == 64, "Metal sizeof(PagedAttentionParams) expected to be 64 bytes.");
constant static const uint kAlignmentBytes = 64;
constant static const uint kAlignmentMask = kAlignmentBytes - 1;
constant static const float kEpsilonForZeroGuard = 1e-9f;
#endif
