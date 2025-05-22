#!/usr/bin/env python3
"""Debug script to verify what data the Metal kernel receives vs what Python sends."""

import os

import mlx.core as mx

from proxy_attention_lab.ops import paged_attention


def validate_metal_input(array, name):
    """Validates MLX array state before passing to Metal kernel."""
    print(f"\n  Validating '{name}' array:")
    print(f"    Shape: {array.shape}")
    print(f"    Dtype: {array.dtype}")
    print(f"    Size: {array.size}")

    # Sample some values
    if array.size > 0:
        flat = array.flatten()
        sample_size = min(8, flat.size)
        print(f"    Sample values (first {sample_size}): {[flat[i].item() for i in range(sample_size)]}")
    return array


# Set debug mode
os.environ["PAL_DEBUG_BUFFERS"] = "1"

# Set random seed for reproducibility
mx.random.seed(42)

# Test parameters matching the failing case
batch_size = 1
history_len = 64
num_q_heads = 2
num_kv_heads = 2
head_dim = 32
dtype = mx.float16
num_queries = 2  # Decode mode
block_size = 8

print(
    f"Test parameters: batch_size={batch_size}, history_len={history_len}, "
    f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
)

# Generate random inputs
queries = mx.random.uniform(-1.0, 1.0, shape=(batch_size, num_queries, num_q_heads, head_dim), dtype=dtype)
keys = mx.random.uniform(-1.0, 1.0, shape=(batch_size, history_len, num_kv_heads, head_dim), dtype=dtype)
values = mx.random.uniform(-1.0, 1.0, shape=(batch_size, history_len, num_kv_heads, head_dim), dtype=dtype)

# Reshape for PAL format
pal_queries = queries.reshape(batch_size * num_queries, num_q_heads, head_dim)

# Apply MLX best practices for array preparation
print("\n=== APPLYING MLX BEST PRACTICES FOR ARRAY PREPARATION ===")

# Ensure pal_queries is fully evaluated and synchronized
print("\nPreparing pal_queries...")
mx.eval(pal_queries)
# Make contiguous as per best practices
pal_queries = mx.contiguous(pal_queries)
mx.eval(pal_queries)
mx.synchronize()
print("  Completed array preparation for pal_queries")

# Print what Python expects for first 64 values
print("\nPython expects queries_in to contain:")
queries_flat = pal_queries.reshape(-1)
for i in range(64):
    if i % 8 == 0:
        print(f"\n  [{i:2d}-{i + 7:2d}]:", end="")
    print(f" {queries_flat[i].item():8.5f}", end="")
print()

# Create K cache pool
max_pages_per_seq = (history_len + block_size - 1) // block_size
k_cache_pool = mx.zeros((max_pages_per_seq, block_size, num_kv_heads, head_dim), dtype=dtype)

# Fill K cache pool page by page
for seq_idx in range(batch_size):
    for kv_head_idx in range(num_kv_heads):
        for token_idx in range(history_len):
            page_idx = token_idx // block_size
            token_in_page = token_idx % block_size
            k_cache_pool[page_idx, token_in_page, kv_head_idx, :] = keys[seq_idx, token_idx, kv_head_idx, :]

# Apply MLX best practices for k_cache_pool
print("\nPreparing k_cache_pool...")
mx.eval(k_cache_pool)
mx.synchronize()
print("  Completed array preparation for k_cache_pool")

# Use the properly prepared k_cache_pool
k_cache_pool_contig = k_cache_pool

k_cache_pool_flat = k_cache_pool_contig.reshape(-1)
mx.eval(k_cache_pool_flat)
mx.synchronize()

# Print what Python expects for K cache at KV-head 1 offset
kv_head_1_offset = 1 * block_size * head_dim
print(f"\nPython expects k_cache_pool_in at KV-head 1 offset ({kv_head_1_offset}) to contain:")
for i in range(64):
    if i % 8 == 0:
        print(f"\n  [{i:2d}-{i + 7:2d}]:", end="")
    print(f" {k_cache_pool_flat[kv_head_1_offset + i].item():8.5f}", end="")
print()

# Create V cache pool (not used by debug kernel but needed for API)
v_cache_pool = mx.zeros_like(k_cache_pool_contig)

# Apply MLX best practices for v_cache_pool
print("\nPreparing v_cache_pool...")
mx.eval(v_cache_pool)
mx.synchronize()
print("  Completed array preparation for v_cache_pool")

# Create page table
page_table = mx.arange(max_pages_per_seq, dtype=mx.uint32).reshape(1, -1)

# Other inputs
sequence_lengths = mx.full((batch_size,), history_len, dtype=mx.int32)
query_to_seq_map = mx.zeros((batch_size * num_queries,), dtype=mx.int32)
query_token_offset = mx.array([history_len + i for i in range(num_queries)], dtype=mx.int32)

# Ensure all arrays are evaluated
mx.eval(sequence_lengths, query_to_seq_map, query_token_offset)

# Debug print dtypes
print("\nDtypes check:")
print(f"  sequence_lengths.dtype: {sequence_lengths.dtype}")
print(f"  query_to_seq_map.dtype: {query_to_seq_map.dtype}")
print(f"  query_token_offset.dtype: {query_token_offset.dtype}")

# Validate all main input arrays before passing to Metal kernel
print("\n=== VALIDATING ALL INPUTS BEFORE PASSING TO METAL KERNEL ===")
validate_metal_input(pal_queries, "pal_queries")
validate_metal_input(k_cache_pool_contig, "k_cache_pool")
validate_metal_input(v_cache_pool, "v_cache_pool")

# Final global synchronization before calling C++ op (as per Advisor G's instructions)
print("\nPerforming final global synchronization before C++ call...")
mx.synchronize()

print("\nCalling paged_attention with debug kernel...")
# This will output debug buffer with Metal kernel's view
debug_output = paged_attention(
    queries=pal_queries,
    k_cache_pool=k_cache_pool_contig,  # Use contiguous version
    v_cache_pool=v_cache_pool,  # Already contiguous
    page_table=page_table,
    sequence_lengths=sequence_lengths,
    query_to_seq_map=query_to_seq_map,
    query_token_offset=query_token_offset,
    is_prefill=False,
)

# The debug kernel writes to the output buffer
# We need to eval it to get the results
print("\n" + "=" * 80)
print("ANALYZING DEBUG OUTPUT FROM METAL KERNEL")
print("=" * 80)

# Eval the debug output to get the results
mx.eval(debug_output)
mx.synchronize()

# The debug kernel writes 128 floats:
# [0:3] = test pattern (123.456, -789.012, 3.14159, -2.71828)
# [4:63] = first 60 values from queries_in as seen by Metal
# [64:128] = first 64 values from k_cache_pool_in at KV-head 1 offset as seen by Metal
debug_flat = debug_output.reshape(-1)

print("\nDebug test pattern (should be 123.456, -789.012, 3.14159, -2.71828):")
print(f"  {debug_flat[0].item()}, {debug_flat[1].item()}, {debug_flat[2].item()}, {debug_flat[3].item()}")

print("\nMetal kernel sees queries_in containing:")
for i in range(60):
    if i % 8 == 0:
        print(f"\n  [{i:2d}-{i + 7:2d}]:", end="")
    print(f" {debug_flat[i + 4].item():8.5f}", end="")
print()

print(f"\nMetal kernel sees k_cache_pool_in at KV-head 1 offset ({kv_head_1_offset}) containing:")
for i in range(64):
    if i % 8 == 0:
        print(f"\n  [{i:2d}-{i + 7:2d}]:", end="")
    print(f" {debug_flat[64 + i].item():8.5f}", end="")
print()

# Compare
print("\n" + "=" * 80)
print("COMPARISON RESULTS:")
print("=" * 80)

queries_match = True
for i in range(60):  # Only check first 60 values
    python_val = queries_flat[i].item()
    metal_val = debug_flat[i + 4].item()  # Shifted by 4 due to test pattern
    if abs(python_val - metal_val) > 1e-5:
        if queries_match:
            print(f"\nQueries mismatch at index {i}: Python={python_val:.6f}, Metal={metal_val:.6f}")
        queries_match = False

if queries_match:
    print("\n✓ Queries match perfectly!")
else:
    print("\n✗ Queries DO NOT match - Metal kernel sees different data!")

k_cache_match = True
for i in range(64):
    python_val = k_cache_pool_flat[kv_head_1_offset + i].item()
    metal_val = debug_flat[64 + i].item()
    if abs(python_val - metal_val) > 1e-5:
        if k_cache_match:
            print(f"\nK-cache mismatch at index {i}: Python={python_val:.6f}, Metal={metal_val:.6f}")
        k_cache_match = False

if k_cache_match:
    print("\n✓ K-cache matches perfectly!")
else:
    print("\n✗ K-cache DOES NOT match - Metal kernel sees different data!")
