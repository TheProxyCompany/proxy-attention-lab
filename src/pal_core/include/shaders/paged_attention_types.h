// Add a comment to ensure exact match between C++ and Metal
// IMPORTANT: This struct must have the same layout in both C++ and Metal
// All members are 4-byte aligned
struct PagedAttentionParams {
    uint32_t num_q_heads;                  // offset 0
    uint32_t num_kv_heads;                 // offset 4
    uint32_t head_dim;                     // offset 8
    uint32_t tokens_per_page;              // offset 12
    float    scale;                        // offset 16
    uint32_t max_logical_blocks_per_seq;   // offset 20
    uint32_t num_physical_pages_in_pool;   // offset 24
    uint32_t num_sequences_in_batch;       // offset 28
    uint32_t actual_threads_per_item_group; // offset 32
    uint32_t total_items_in_dispatch;      // offset 36
    // Total size: 40 bytes
};
