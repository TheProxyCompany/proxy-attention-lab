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

from proxy_attention_lab.ops import fill_kv_pages, paged_attention
from proxy_attention_lab.pal_core import get_optimal_tile_size as calculate_page_size

__all__ = ["calculate_page_size", "fill_kv_pages", "paged_attention"]
