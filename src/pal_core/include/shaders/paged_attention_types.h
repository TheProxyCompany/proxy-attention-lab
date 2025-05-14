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
  float    scale;                         // Scale factor for attention scores
  uint32_t max_logical_blocks_per_seq;    // Maximum logical blocks per sequence
  uint32_t num_physical_pages_in_pool;    // Number of physical pages in pool
  uint32_t num_sequences_in_batch;        // Number of sequences in batch
  uint32_t actual_threads_per_item_group; // Actual threads in each threadgroup
  uint32_t total_items_in_dispatch;       // Total items being dispatched
};

#ifdef __METAL_VERSION__ // Only apply this check when compiling with Metal
// Use C++11 static_assert, which MSL (C++14 based) should support
static_assert(sizeof(PagedAttentionParams) == 48,
              "sizeof(PagedAttentionParams) mismatch between Metal and expected "
              "size (48 bytes). Check struct definition and padding.");
#endif
