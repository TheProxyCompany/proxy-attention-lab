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
"""Proxy Attention Lab (PAL): A laboratory for experimenting with attention primitives in MLX."""

from proxy_attention_lab.ops import fill_kv_pages, get_k_cache_stripe_size, paged_attention
from proxy_attention_lab.pal_core import (
    get_k_cache_shape,
    get_optimal_page_size,
    get_v_cache_shape,
)

__all__ = [
    "fill_kv_pages",
    "get_k_cache_shape",
    "get_k_cache_stripe_size",
    "get_optimal_page_size",
    "get_v_cache_shape",
    "paged_attention",
]
