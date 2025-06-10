# # Copyright 2025 The Proxy Company. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """Tests for GQA (Grouped Query Attention) and MQA (Multi-Query Attention).

# This module contains tests that verify the correct behavior of the paged attention
# operation when using Grouped Query Attention (GQA) and Multi-Query Attention (MQA)
# configurations, where the number of query heads can differ from the number of key-value heads.
# """

# import logging

# import mlx.core as mx
# import pytest

# from proxy_attention_lab import paged_attention

# logger = logging.getLogger(__name__)

# @pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
# @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
# def test_fetch_k_vector_from_multiple_kv_heads(head_dim, dtype) -> None:
#     """Test GQA with multiple Q heads mapping to KV heads.

#     This test verifies that in Grouped Query Attention (GQA) mode, multiple query heads
#     correctly map to their corresponding KV heads and compute accurate dot products.
#     """
#     num_tokens = 1
#     num_q_heads = 2
#     cfg_tokens_per_page = 16
#     cfg_num_kv_heads = 2
#     cfg_head_dim = head_dim
#     cfg_max_logical_pages_per_seq_in_pagetable = 2
#     py_queries = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=dtype)
#     py_queries[0, 0, :] = 100.0
#     py_queries[0, 1, :] = 200.0
#     num_physical_pages = 1
#     k_cache_shape = (num_physical_pages, cfg_num_kv_heads, cfg_tokens_per_page, cfg_head_dim)
#     py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)
#     token_slot = 5
#     for i in range(cfg_head_dim):
#         py_k_cache_pool[0, 0, token_slot, i] = float(i + 1)
#         py_k_cache_pool[0, 1, token_slot, i] = float(i + 5)
#     py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

#     # Set up V-cache pool with distinct values for each K-vector position
#     # Values for KV head 0 (used by Q head 0)
#     py_v_cache_pool[0, 0, token_slot, :] = mx.array([10.0, 11.0, 12.0, 13.0], dtype=dtype)
#     # Values for KV head 1 (used by Q head 1)
#     py_v_cache_pool[0, 1, token_slot, :] = mx.array([20.0, 21.0, 22.0, 23.0], dtype=dtype)

#     py_page_table = mx.array(
#         [
#             [0, 99],
#         ],
#         dtype=mx.uint32,
#     )
#     assert py_page_table.shape == (1, cfg_max_logical_pages_per_seq_in_pagetable)
#     py_sequence_lengths = mx.array([64], dtype=mx.int32)

#     output_arr = paged_attention(
#         py_queries,
#         py_k_cache_pool,
#         py_v_cache_pool,
#         py_page_table,
#         py_sequence_lengths
#     )
#     mx.eval(output_arr)

#     # Expected dot products for GQA mapping
#     # For token 0, q_head 0 -> k_head 0:
#     # Q[0,0,:] = [100.0, 100.0, 100.0, 100.0], K = [1.0, 2.0, 3.0, 4.0]
#     # Dot product = 100.0 * 1.0 + 100.0 * 2.0 + 100.0 * 3.0 + 100.0 * 4.0 = 1000.0
#     # Scaled = 1000.0 * py_scale = 1000.0 / 2.0 = 500.0

#     # For token 0, q_head 1 -> k_head 1:
#     # Q[0,1,:] = [200.0, 200.0, 200.0, 200.0], K = [5.0, 6.0, 7.0, 8.0]
#     # Dot product = 200.0 * 5.0 + 200.0 * 6.0 + 200.0 * 7.0 + 200.0 * 8.0 = 5200.0
#     # Scaled = 5200.0 * py_scale = 5200.0 / 2.0 = 2600.0

#     # Output is now full attention format [num_q_threads, cfg_head_dim]
#     # For 3D queries, shape is [num_tokens * num_q_heads, cfg_head_dim]
#     total_items = num_tokens * num_q_heads
#     expected_output_shape = (total_items, cfg_head_dim)

#     # Calculate expected V-output for each query head
#     # Since we only have one history token per query, each softmax prob is 1.0
#     # Therefore V-output for each query head is exactly the corresponding V-vector

#     # For Q head 0 -> KV head 0: V-vector should be [10, 11, 12, 13]
#     expected_v_head0 = mx.array([10.0, 11.0, 12.0, 13.0], dtype=dtype)

#     # For Q head 1 -> KV head 1: V-vector should be [20, 21, 22, 23]
#     expected_v_head1 = mx.array([20.0, 21.0, 22.0, 23.0], dtype=dtype)

#     # Combine into expected V-output array
#     expected_v_output = mx.stack([expected_v_head0, expected_v_head1])

#     logger.info(f"Test: {test_fetch_k_vector_from_multiple_kv_heads.__name__} (dtype={dtype})")
#     logger.info(f"  GQA configuration: num_q_heads={num_q_heads}, num_kv_heads={cfg_num_kv_heads}")
#     logger.info(f"  Expected V output: {expected_v_output}")
#     logger.info(f"  Actual V output: {output_arr}")

#     # Verify results
#     assert output_arr.shape == expected_output_shape, (
#         f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
#     )
#     assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
#     # Check V output vectors
#     assert mx.allclose(output_arr, expected_v_output, atol=1e-2, rtol=1e-2), (
#         "V output vectors do not match expected values for GQA mapping"
#     )



# @pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
# @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
# def test_mqa_kv_head_selection(head_dim, dtype) -> None:
#     """Test Multi-Query Attention (MQA) KV head selection.

#     This test verifies that the kernel correctly maps query heads to KV heads
#     when there are fewer query heads than KV heads, ensuring each query head
#     attends to the correct KV head according to the mapping logic.

#     Specifically, with num_q_heads=1 and num_kv_heads=2, the test confirms
#     that queries use KV head 0 as specified in the kernel's MQA logic.
#     """
#     # MQA configuration: fewer query heads than KV heads
#     num_tokens = 1
#     num_q_heads = 1  # Only one query head
#     cfg_tokens_per_page = 16
#     cfg_head_dim = head_dim
#     cfg_num_kv_heads = 2  # Two KV heads

#     # Create 3D queries with shape [num_tokens, num_q_heads, cfg_head_dim]
#     py_queries = mx.zeros((num_tokens, num_q_heads, cfg_head_dim), dtype=dtype)
#     # Q-vector for the single query head
#     py_queries[0, 0, :] = mx.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)

#     # Create K-cache pool with different K-vectors in each KV head
#     num_physical_pages = 1
#     k_cache_shape = (num_physical_pages, cfg_num_kv_heads, cfg_tokens_per_page, cfg_head_dim)
#     py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)

#     # K-vector for KV head 0 - this is the one that should be used
#     py_k_cache_pool[0, 0, 0, :] = mx.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
#     # K-vector for KV head 1 - should NOT be used by the single query head
#     py_k_cache_pool[0, 1, 0, :] = mx.array([2.0, 2.0, 2.0, 2.0], dtype=dtype)

#     py_v_cache_pool = mx.zeros_like(py_k_cache_pool)
#     # Set up V-cache pool with distinct values for each KV head
#     # For KV head 0 (which will be used in MQA)
#     py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
#     # For KV head 1 (which would be incorrect to use)
#     py_v_cache_pool[0, 1, 0, :] = mx.array([50.0, 60.0, 70.0, 80.0], dtype=dtype)

#     py_page_table = mx.array([[0]], dtype=mx.uint32)  # Logical block 0 -> Physical page 0
#     py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)

#     mx.eval(py_queries)
#     mx.eval(py_k_cache_pool)
#     mx.eval(py_v_cache_pool)
#     mx.eval(py_page_table)
#     mx.eval(py_sequence_lengths)

#     # Call the kernel
#     output_arr = paged_attention(
#         py_queries,
#         py_k_cache_pool,
#         py_v_cache_pool,
#         py_page_table,
#         py_sequence_lengths
#     )
#     mx.eval(output_arr)

#     # Calculate expected results
#     # Q=[1,2,3,4] with K=[1,1,1,1] from kv_head=0 gives dot product = 10

#     # Output is now full attention format [num_q_threads, cfg_head_dim]
#     total_items = num_tokens * num_q_heads
#     expected_output_shape = (total_items, cfg_head_dim)

#     # Since we only have one history token, the softmax prob is 1.0
#     # Therefore, expected V output should be exactly the V vector from KV head 0
#     expected_v_output = py_v_cache_pool[0, 0, 0, :].reshape(1, cfg_head_dim)

#     # Incorrect V output would be using KV head 1's V vector
#     incorrect_v_output = py_v_cache_pool[0, 0, 1, :].reshape(1, cfg_head_dim)

#     logger.info(f"Test: {test_mqa_kv_head_selection.__name__} (dtype={dtype})")
#     logger.info(f"  MQA configuration: num_q_heads={num_q_heads}, num_kv_heads={cfg_num_kv_heads}")
#     logger.info(f"  Q = {py_queries[0, 0, :]}, K (KV head 0) = {py_k_cache_pool[0, 0, 0, :]}")
#     logger.info(f"  Correct V (KV head 0) = {py_v_cache_pool[0, 0, 0, :]}")
#     logger.info(f"  Incorrect V (KV head 1) = {py_v_cache_pool[0, 0, 1, :]}")
#     logger.info(f"  Actual V output = {output_arr}")

#     # Verify results
#     assert output_arr.shape == expected_output_shape, (
#         f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
#     )
#     assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
#     # Verify that the kernel is correctly using KV head 0 for the query by checking the V output
#     assert mx.allclose(output_arr, expected_v_output, atol=1e-2, rtol=1e-2), (
#         "MQA is not correctly using KV head 0 for the query"
#     )
#     # Also explicitly verify we're not getting the incorrect V-vector from KV head 1
#     assert not mx.allclose(output_arr, incorrect_v_output, atol=1e-2, rtol=1e-2), (
#         "MQA is incorrectly using KV head 1 instead of KV head 0"
#     )


# @pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
# @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
# def test_mqa_multi_token_kv_head_selection_2d_query(head_dim, dtype) -> None:
#     """Test MQA with multi-token KV head selection using 2D queries.

#     This test verifies consistent KV head selection behavior with 2D queries
#     in MQA mode, ensuring all queries properly select KV head 0 regardless of
#     token position.
#     """
#     # Test configuration
#     num_tokens = 5  # Multiple tokens to test consistent KV-head selection
#     cfg_tokens_per_page = 16
#     cfg_num_kv_heads = 4  # Multiple KV heads
#     cfg_head_dim = head_dim
#     # Create 2D queries with shape [num_tokens, cfg_head_dim]
#     # For 2D queries, the C++ primitive sets params->num_q_heads = 1 internally
#     py_queries = mx.array([[1.0] * cfg_head_dim] * num_tokens, dtype=dtype)

#     # Create K-cache pool with zeros
#     num_physical_pages = 1
#     k_cache_shape = (num_physical_pages, cfg_num_kv_heads, cfg_tokens_per_page, cfg_head_dim)
#     py_k_cache_pool = mx.zeros(k_cache_shape, dtype=dtype)
#     py_v_cache_pool = mx.zeros_like(py_k_cache_pool)

#     # Add V-cache with distinct values for each token position
#     # All queries use KV-head 0 in MQA mode when queries are 2D
#     py_v_cache_pool[0, 0, 0, :] = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)

#     py_page_table = mx.array([[0]], dtype=mx.uint32)  # Simple page table
#     py_sequence_lengths = mx.array([cfg_tokens_per_page], dtype=mx.int32)  # Plenty of tokens in the sequence

#     mx.eval(py_queries)
#     mx.eval(py_k_cache_pool)
#     mx.eval(py_v_cache_pool)
#     mx.eval(py_page_table)
#     mx.eval(py_sequence_lengths)

#     # Call the kernel with our debug version
#     output_arr = paged_attention(
#         py_queries,
#         py_k_cache_pool,
#         py_v_cache_pool,
#         py_page_table,
#         py_sequence_lengths
#     )
#     mx.eval(output_arr)

#     # Output is now full attention format [num_tokens, cfg_head_dim]
#     expected_output_shape = (num_tokens, cfg_head_dim)

#     # For each query token, we expect the V-vector from KV head 0
#     # Since we only have one history token per query item, softmax prob is 1.0
#     # So the output should be exactly the V-vector from KV head 0
#     # Create a stack of the same V-vector for each token
#     single_v = mx.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)
#     expected_v_output = mx.stack([single_v] * num_tokens)

#     logger.info(f"Test: {test_mqa_multi_token_kv_head_selection_2d_query.__name__} (dtype={dtype})")
#     logger.info(f"  MQA configuration: 2D queries, num_kv_heads={cfg_num_kv_heads}")
#     logger.info(f"  Number of tokens: {num_tokens}")
#     logger.info(f"  Expected V output: {expected_v_output[0]} (repeated for each token)")
#     logger.info(f"  Actual output shape: {output_arr.shape}")

#     # Verify results
#     assert output_arr.shape == expected_output_shape, (
#         f"Output shape {output_arr.shape} does not match expected {expected_output_shape}"
#     )
#     assert output_arr.dtype == dtype, f"Output dtype {output_arr.dtype} does not match {dtype}"
#     # Check values match expected V-vector from correct KV head
#     assert mx.allclose(output_arr, expected_v_output, atol=1e-2, rtol=1e-2), (
#         "MQA with 2D queries is not correctly selecting KV head 0"
#     )
