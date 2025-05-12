struct PagedAttentionParams {
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t tokens_per_page;
    float    scale;
    uint32_t max_logical_blocks_per_seq;
};
