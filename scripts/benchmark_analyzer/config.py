"""Configuration constants for benchmark analyzer."""

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
COL_PARAMS_STR = "params_str"
COL_MEAN_LATENCY = "mean_latency_ms"
COL_THROUGHPUT = "throughput_items_per_sec"
