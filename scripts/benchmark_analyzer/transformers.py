"""Data transformation utilities for benchmark analyzer."""

from __future__ import annotations

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
    PARSER_MAP = {
        "BM_PAL_LatencyVsSeqLen": lambda p: (
            {"seq_len": _parse_int(p.split("/")[-1]), "num_query_items": config.DEFAULT_NUM_QUERY_ITEMS} if p else {}
        ),
        "test_pal_latency_vs_seq_len": lambda p: (
            {"seq_len": _parse_int(p.split("/")[-1]), "num_query_items": config.DEFAULT_NUM_QUERY_ITEMS} if p else {}
        ),
        "BM_SDPA_LatencyVsNumItems": lambda p: {"batch_size": _parse_int(p.split("/")[0])} if p else {},
        "test_sdpa_latency_vs_query_items": lambda p: {"batch_size": _parse_int(p.split("/")[0])} if p else {},
        "BM_PAL_LatencyVsHeadDim": lambda p: {"head_dim": _parse_int(p.split("/")[-1])} if p else {},
        "test_pal_latency_vs_head_dim": lambda p: {"head_dim": _parse_int(p.split("/")[-1])} if p else {},
        "BM_SDPA_LatencyVsHeadDim": lambda p: {"head_dim": _parse_int(p.split("/")[-1])} if p else {},
        "test_sdpa_latency_vs_head_dim": lambda p: {"head_dim": _parse_int(p.split("/")[-1])} if p else {},
    }

    for idx, row in df.iterrows():
        base = row[config.COL_BENCHMARK_NAME_BASE]
        params = row[config.COL_PARAMS_STR]
        parser = PARSER_MAP.get(base)
        if parser:
            updates = parser(params)
            for key, value in updates.items():
                df.at[idx, key] = value
    # Effective items for comparisons
    df["effective_items"] = df["num_query_items"].fillna(df["batch_size"])
    return df
