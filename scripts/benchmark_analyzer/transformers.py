"""Data transformation utilities for benchmark analyzer."""

from __future__ import annotations

import re

import pandas as pd

from . import config


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_and_normalize_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich benchmark DataFrame with normalized parameters."""
    df = df.copy()
    df["seq_len"] = pd.NA
    df["head_dim"] = pd.NA
    df["num_query_items"] = pd.NA
    df["batch_size"] = pd.NA
    for idx, row in df.iterrows():
        base = row[config.COL_BENCHMARK_NAME_BASE]
        params = row[config.COL_PARAMS_STR]
        source = row[config.COL_SOURCE]
        if base in {"BM_PAL_LatencyVsSeqLen", "test_pal_latency_vs_seq_len"}:
            if params:
                df.at[idx, "seq_len"] = _parse_int(params.split("/")[-1])
                df.at[idx, "num_query_items"] = config.DEFAULT_NUM_QUERY_ITEMS
        elif base in {"BM_SDPA_LatencyVsNumItems", "test_sdpa_latency_vs_query_items"}:
            if params:
                df.at[idx, "batch_size"] = _parse_int(params.split("/")[0])
        elif base in {"BM_PAL_LatencyVsHeadDim", "test_pal_latency_vs_head_dim"}:
            if params:
                df.at[idx, "head_dim"] = _parse_int(params.split("/")[-1])
        elif base in {"BM_SDPA_LatencyVsHeadDim", "test_sdpa_latency_vs_head_dim"}:
            if params:
                df.at[idx, "head_dim"] = _parse_int(params.split("/")[-1])
    # Effective items for comparisons
    df["effective_items"] = df["num_query_items"].fillna(df["batch_size"])
    return df
