"""Configuration constants for benchmark analyzer."""

# Default values for benchmark parameters
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_Q_HEADS = 1
DEFAULT_NUM_KV_HEADS = 1
DEFAULT_SEQ_LEN = 128
DEFAULT_NUM_QUERY_ITEMS = 64
DEFAULT_TOKENS_PER_PAGE = 64
DEFAULT_NUM_SEQUENCES_IN_BATCH = 1
DEFAULT_BATCH_SIZE = 64

# Column names used across the package
COL_BENCHMARK_NAME_BASE = "benchmark_name_base"
COL_SOURCE = "source"
COL_KERNEL_NAME = "kernel_name"  # Added for extensibility
COL_PARAMS_STR = "params_str"
COL_MEAN_LATENCY = "mean_latency_ms"
COL_THROUGHPUT = "throughput_items_per_sec"

# Model configuration parameters
MODEL_CONFIG_PARAMETERS = {
    "Llama3_70B_Sim": {
        "num_q_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 1024,
        # PAL-specific parameters
        "tokens_per_page": 64,
        "num_sequences_in_batch": 1,
        "pal_num_query_items": 64 * 64,  # 64 tokens in batch * 64 query heads
        # SDPA-specific parameters
        "sdpa_batch_size": 4,
    },
    "Qwen_8B_Sim": {
        "num_q_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "seq_len": 1024,
        # PAL-specific parameters
        "tokens_per_page": 64,
        "num_sequences_in_batch": 1,
        "pal_num_query_items": 64 * 32,  # 64 tokens in batch * 32 query heads
        # SDPA-specific parameters
        "sdpa_batch_size": 4,
    },
    "Qwen2.5_72B_Sim": {
        "num_q_heads": 128,
        "num_kv_heads": 8,
        "head_dim": 128,
        "seq_len": 1024,
        # PAL-specific parameters
        "tokens_per_page": 64,
        "num_sequences_in_batch": 1,
        "pal_num_query_items": 64 * 128,  # 64 tokens in batch * 128 query heads
        # SDPA-specific parameters
        "sdpa_batch_size": 4,
    },
}
