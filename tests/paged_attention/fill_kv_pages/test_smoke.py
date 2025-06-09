# Copyright 2025 The Proxy Company. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic smoke test for fill_kv_pages functionality."""

import logging

import mlx.core as mx
import pytest

from proxy_attention_lab import fill_kv_pages

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_fill_kv_pages_smoke(dtype: mx.Dtype) -> None:
    """
    Smoke test for fill_kv_pages: verifies the function runs on minimal, valid inputs and produces outputs of correct shape and dtype.
    """
    num_new_tokens = 2
    num_sequences_in_batch = 2
    num_kv_heads = 8
    head_dim = 64
    tokens_per_page = 16
    num_physical_pages = 10
    max_logical_pages_per_seq = 4

    new_keys = mx.random.normal([num_new_tokens, num_kv_heads, head_dim], dtype=dtype)
    new_values = mx.random.normal([num_new_tokens, num_kv_heads, head_dim], dtype=dtype)
    global_key_pool = mx.zeros([num_physical_pages, tokens_per_page, num_kv_heads, head_dim], dtype=dtype)
    global_value_pool = mx.zeros([num_physical_pages, tokens_per_page, num_kv_heads, head_dim], dtype=dtype)
    pt_data = mx.arange(num_sequences_in_batch * max_logical_pages_per_seq, dtype=mx.uint32) % num_physical_pages
    page_table = mx.array(pt_data.reshape(num_sequences_in_batch, max_logical_pages_per_seq))
    current_token_write_positions = mx.array(mx.zeros(num_new_tokens, dtype=mx.int32))
    query_to_seq_map = mx.array(mx.arange(num_new_tokens, dtype=mx.uint32))

    mx.eval(
        new_keys,
        new_values,
        global_key_pool,
        global_value_pool,
        page_table,
        current_token_write_positions,
        query_to_seq_map,
    )

    updated_k_pool, updated_v_pool = fill_kv_pages(
        new_keys,
        new_values,
        global_key_pool,
        global_value_pool,
        page_table,
        current_token_write_positions,
        query_to_seq_map,
    )

    mx.eval(updated_k_pool, updated_v_pool)

    assert mx.sum(mx.isinf(updated_k_pool)) == 0, "updated_k_pool contains inf values"
    assert updated_k_pool.shape == global_key_pool.shape
    assert updated_k_pool.dtype == global_key_pool.dtype
    assert updated_v_pool.shape == global_value_pool.shape
    assert updated_v_pool.dtype == global_value_pool.dtype
