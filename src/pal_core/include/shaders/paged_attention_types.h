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
#include <type_traits>  // For std::is_standard_layout_v
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
  uint32_t max_accum_tile_runtime;        // Runtime V-accum tile size
  uint32_t tile_size_T_runtime;           // Runtime tile size T for history processing
  float    log_exp_min_clamp;             // Minimum value for exponent in exp function
  uint32_t pad_floats_per_row;            // Padding floats per row for K/V tiles
};

// --- Assertions ---
#ifndef __METAL_VERSION__ // C++ side

static_assert(std::is_standard_layout_v<PagedAttentionParams>,
              "PagedAttentionParams must be a standard-layout type.");
static_assert(alignof(PagedAttentionParams) == 16,
              "PagedAttentionParams must have 16-byte alignment.");
// 10 uint32_t (40 bytes) + 1 float (4 bytes) = 44 data bytes.
// alignas(16) means total size is 48, as it's padded to multiple of 16.
static_assert(sizeof(PagedAttentionParams) == 48,
              "C++ sizeof(PagedAttentionParams) expected to be 48 bytes.");

#else // __METAL_VERSION__ (Metal side)
static_assert(sizeof(PagedAttentionParams) == 48,
              "Metal sizeof(PagedAttentionParams) expected to be 48 bytes.");
#endif
