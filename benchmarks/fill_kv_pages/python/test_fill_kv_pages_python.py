import mlx.core as mx
import pytest

from proxy_attention_lab import fill_kv_pages

# A memory alignment of 16 bytes is standard for Metal.
MEMORY_ALIGNMENT_BYTES = 16


class TestFillKVPagesBenchmarks:
    """Benchmarks for fill_kv_pages operation."""

    @pytest.mark.parametrize("num_new_tokens", [1, 4, 8, 16, 32, 64, 128])
    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_fill_multiple_tokens_single_sequence_benchmark(self, benchmark, num_new_tokens, dtype):
        """Benchmark filling multiple tokens from a single sequence."""
        # Parameters
        num_kv_heads = 8
        head_dim = 128
        tokens_per_page = 16
        num_physical_pages = max(4, (num_new_tokens // tokens_per_page) + 2)
        elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

        # Source data
        new_keys = mx.random.normal((num_new_tokens, num_kv_heads, head_dim)).astype(dtype)
        new_values = mx.random.normal((num_new_tokens, num_kv_heads, head_dim)).astype(dtype)

        # Destination caches
        global_key_pool = mx.zeros(
            (
                num_physical_pages,
                num_kv_heads,
                head_dim // elements_per_thread,
                tokens_per_page,
                elements_per_thread,
            ),
            dtype=dtype,
        )
        global_value_pool = mx.zeros(
            (
                num_physical_pages,
                num_kv_heads,
                head_dim,
                tokens_per_page,
            ),
            dtype=dtype,
        )

        # Paging metadata
        num_logical_blocks = (num_new_tokens + tokens_per_page - 1) // tokens_per_page
        page_table = mx.arange(num_logical_blocks, dtype=mx.uint32).reshape(1, -1)
        current_token_write_positions = mx.arange(num_new_tokens, dtype=mx.int32)
        query_to_seq_map = mx.zeros(num_new_tokens, dtype=mx.uint32)

        # Ensure arrays are evaluated before benchmarking
        mx.eval(new_keys, new_values, global_key_pool, global_value_pool, page_table)

        def run_fill():
            updated_k, updated_v = fill_kv_pages(
                new_keys=new_keys,
                new_values=new_values,
                global_key_pool=global_key_pool,
                global_value_pool=global_value_pool,
                page_table=page_table,
                current_token_write_positions=current_token_write_positions,
                query_to_seq_map=query_to_seq_map,
            )
            mx.eval(updated_k, updated_v)
            return updated_k, updated_v

        benchmark(run_fill)

    @pytest.mark.parametrize("num_sequences", [1, 2, 4, 8, 16, 32, 64])
    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_fill_single_token_multiple_sequences_benchmark(self, benchmark, num_sequences, dtype):
        """Benchmark filling single tokens from multiple sequences (batched decode)."""
        # Parameters
        num_kv_heads = 8
        head_dim = 128
        tokens_per_page = 16
        num_physical_pages = max(4, num_sequences // 2)  # Some sequences may share pages
        elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

        # Source data: one token per sequence
        new_keys = mx.random.normal((num_sequences, num_kv_heads, head_dim)).astype(dtype)
        new_values = mx.random.normal((num_sequences, num_kv_heads, head_dim)).astype(dtype)

        # Destination caches
        global_key_pool = mx.zeros(
            (
                num_physical_pages,
                num_kv_heads,
                head_dim // elements_per_thread,
                tokens_per_page,
                elements_per_thread,
            ),
            dtype=dtype,
        )
        global_value_pool = mx.zeros(
            (
                num_physical_pages,
                num_kv_heads,
                head_dim,
                tokens_per_page,
            ),
            dtype=dtype,
        )

        # Paging metadata: distribute sequences across pages
        page_table = mx.zeros((num_sequences, 1), dtype=mx.uint32)
        for i in range(num_sequences):
            page_table[i, 0] = i % num_physical_pages

        # Write positions: spread tokens across slots
        current_token_write_positions = mx.arange(num_sequences, dtype=mx.int32) % tokens_per_page
        query_to_seq_map = mx.arange(num_sequences, dtype=mx.uint32)

        # Ensure arrays are evaluated before benchmarking
        mx.eval(new_keys, new_values, global_key_pool, global_value_pool, page_table)

        def run_fill():
            updated_k, updated_v = fill_kv_pages(
                new_keys=new_keys,
                new_values=new_values,
                global_key_pool=global_key_pool,
                global_value_pool=global_value_pool,
                page_table=page_table,
                current_token_write_positions=current_token_write_positions,
                query_to_seq_map=query_to_seq_map,
            )
            mx.eval(updated_k, updated_v)
            return updated_k, updated_v

        benchmark(run_fill)

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("num_kv_heads", [1, 4, 8, 16])
    @pytest.mark.parametrize("dtype", [mx.float16])
    def test_fill_varying_dimensions_benchmark(self, benchmark, head_dim, num_kv_heads, dtype):
        """Benchmark fill operation with varying head dimensions and number of heads."""
        # Fixed parameters
        num_new_tokens = 32
        tokens_per_page = 16
        num_physical_pages = 4
        elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

        # Source data
        new_keys = mx.random.normal((num_new_tokens, num_kv_heads, head_dim)).astype(dtype)
        new_values = mx.random.normal((num_new_tokens, num_kv_heads, head_dim)).astype(dtype)

        # Destination caches
        global_key_pool = mx.zeros(
            (
                num_physical_pages,
                num_kv_heads,
                head_dim // elements_per_thread,
                tokens_per_page,
                elements_per_thread,
            ),
            dtype=dtype,
        )
        global_value_pool = mx.zeros(
            (
                num_physical_pages,
                num_kv_heads,
                head_dim,
                tokens_per_page,
            ),
            dtype=dtype,
        )

        # Paging metadata
        page_table = mx.array([[0, 1]], dtype=mx.uint32)
        current_token_write_positions = mx.arange(num_new_tokens, dtype=mx.int32)
        query_to_seq_map = mx.zeros(num_new_tokens, dtype=mx.uint32)

        # Ensure arrays are evaluated before benchmarking
        mx.eval(new_keys, new_values, global_key_pool, global_value_pool, page_table)

        def run_fill():
            updated_k, updated_v = fill_kv_pages(
                new_keys=new_keys,
                new_values=new_values,
                global_key_pool=global_key_pool,
                global_value_pool=global_value_pool,
                page_table=page_table,
                current_token_write_positions=current_token_write_positions,
                query_to_seq_map=query_to_seq_map,
            )
            mx.eval(updated_k, updated_v)
            return updated_k, updated_v

        benchmark(run_fill)

    @pytest.mark.parametrize("tokens_per_page", [8, 16, 32, 64])
    @pytest.mark.parametrize("dtype", [mx.float16])
    def test_fill_varying_page_size_benchmark(self, benchmark, tokens_per_page, dtype):
        """Benchmark fill operation with varying page sizes."""
        # Fixed parameters
        num_new_tokens = 64
        num_kv_heads = 8
        head_dim = 128
        num_physical_pages = 4
        elements_per_thread = MEMORY_ALIGNMENT_BYTES // dtype.size

        # Source data
        new_keys = mx.random.normal((num_new_tokens, num_kv_heads, head_dim)).astype(dtype)
        new_values = mx.random.normal((num_new_tokens, num_kv_heads, head_dim)).astype(dtype)

        # Destination caches
        global_key_pool = mx.zeros(
            (
                num_physical_pages,
                num_kv_heads,
                head_dim // elements_per_thread,
                tokens_per_page,
                elements_per_thread,
            ),
            dtype=dtype,
        )
        global_value_pool = mx.zeros(
            (
                num_physical_pages,
                num_kv_heads,
                head_dim,
                tokens_per_page,
            ),
            dtype=dtype,
        )

        # Paging metadata
        num_logical_blocks = (num_new_tokens + tokens_per_page - 1) // tokens_per_page
        num_logical_blocks = min(num_logical_blocks, num_physical_pages)
        page_table = mx.arange(num_logical_blocks, dtype=mx.uint32).reshape(1, -1)
        current_token_write_positions = mx.arange(num_new_tokens, dtype=mx.int32)
        query_to_seq_map = mx.zeros(num_new_tokens, dtype=mx.uint32)

        # Ensure arrays are evaluated before benchmarking
        mx.eval(new_keys, new_values, global_key_pool, global_value_pool, page_table)

        def run_fill():
            updated_k, updated_v = fill_kv_pages(
                new_keys=new_keys,
                new_values=new_values,
                global_key_pool=global_key_pool,
                global_value_pool=global_value_pool,
                page_table=page_table,
                current_token_write_positions=current_token_write_positions,
                query_to_seq_map=query_to_seq_map,
            )
            mx.eval(updated_k, updated_v)
            return updated_k, updated_v

        benchmark(run_fill)
