import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_format(json_file: Path) -> str:
    """Detect if the JSON file is from pytest-benchmark or Google Benchmark."""
    with open(json_file) as f:
        data = json.load(f)

    # For Google Benchmark
    if "context" in data and "benchmarks" in data:
        return "google"

    # For pytest-benchmark
    if "machine_info" in data and "benchmarks" in data:
        return "pytest"

    raise ValueError(f"Unknown benchmark format for file: {json_file}")


def parse_google_benchmark(json_file: Path) -> list[dict]:
    """Parse a Google Benchmark JSON file and extract key metrics."""
    with open(json_file) as f:
        data = json.load(f)

    rows = []
    benchmarks = data.get("benchmarks", [])

    for bench in benchmarks:
        # Only process aggregate results for mean
        if bench.get("run_type") == "aggregate" and bench.get("aggregate_name") == "mean":
            name = bench.get("name", "")
            # Extract base name (before first slash)
            base_name_match = re.match(r"(?P<base>[^/]+)", name)
            base_name = base_name_match.group("base") if base_name_match else name

            # Extract params after the base name
            params_str = name[len(base_name) :].lstrip("/") if len(name) > len(base_name) else ""
            # Remove any _mean suffix from the params_str
            params_str = re.sub(r"_mean$", "", params_str)

            # Determine source based on benchmark name
            if base_name.startswith("BM_PAL_"):
                source = "cpp_pal"
            elif base_name.startswith("BM_SDPA_"):
                source = "cpp_sdpa"
            else:
                source = "cpp"  # Default for any other C++ benchmarks

            # Get metrics and convert units
            time_unit = bench.get("time_unit", "ns")
            real_time = bench.get("real_time", 0)

            # Convert to ms for latency (assume ns input)
            mean_latency_ms = real_time / 1_000_000.0 if time_unit == "ns" else None

            # Throughput metrics - ensure we get them from the right fields
            # Items per second is set via SetItemsProcessed in benchmarks
            throughput_items_per_sec = bench.get("items_per_second", None)
            # Bytes per second is set via SetBytesProcessed in benchmarks
            bytes_per_second = bench.get("bytes_per_second", None)
            throughput_gb_per_sec = bytes_per_second / (1024**3) if bytes_per_second else None

            # Debug print to see what benchmarks contain
            if throughput_items_per_sec is None and "LatencyVsNumItems" in base_name:
                print(f"Note: Benchmark {name} has no items_per_second metric despite name suggesting it should.")
                print(f"  Available metrics: {list(bench.keys())}")

            rows.append(
                {
                    "benchmark_name_base": base_name,
                    "full_name": name,
                    "source": source,
                    "mean_latency_ms": mean_latency_ms,
                    "throughput_items_per_sec": throughput_items_per_sec,
                    "throughput_gb_per_sec": throughput_gb_per_sec,
                    "iterations": bench.get("iterations", None),
                    "params_str": params_str,
                }
            )

    return rows


def parse_pytest_benchmark(json_file: Path) -> list[dict]:
    """Parse a pytest-benchmark JSON file and extract key metrics."""
    with open(json_file) as f:
        data = json.load(f)

    rows = []
    benchmarks = data.get("benchmarks", [])

    for bench in benchmarks:
        name = bench.get("name", "")

        # Extract group as base name if available, otherwise parse from name
        group = bench.get("group", "")
        base_name = group if group else name.split("[")[0] if "[" in name else name

        # Determine source based on benchmark name
        if base_name.startswith("test_pal_"):
            source = "python_pal"
        elif base_name.startswith("test_sdpa_"):
            source = "python_sdpa"
        else:
            source = "python"  # Default for any other Python benchmarks

        # Extract params from name (inside brackets)
        params_str = ""
        if "[" in name and "]" in name:
            params_str = name.split("[")[1].split("]")[0]

        # Extract stats
        stats = bench.get("stats", {})
        mean_time_sec = stats.get("mean", 0)
        stddev_sec = stats.get("stddev", 0)

        # Convert to ms
        mean_latency_ms = mean_time_sec * 1000.0
        stddev_latency_ms = stddev_sec * 1000.0

        # Extract model config name if present
        model_config_name_raw = None
        if "model_configs" in base_name and "[" in name and "-model_params" in name:
            config_part = name.split("[")[1].split("-model_params")[0]
            model_config_name_raw = config_part

        # Extract model params if present
        model_params_raw = None
        if "params" in bench and "model_params" in bench["params"]:
            model_params_raw = json.dumps(bench["params"]["model_params"])

        rows.append(
            {
                "benchmark_name_base": base_name,
                "full_name": name,
                "source": source,
                "mean_latency_ms": mean_latency_ms,
                "stddev_latency_ms": stddev_latency_ms,
                "rounds": stats.get("rounds", None),
                "iterations_per_round": stats.get("iterations", None),
                "params_str": params_str,
                "model_config_name_raw": model_config_name_raw,
                "model_params_raw": model_params_raw,
            }
        )

    return rows


def load_results(json_files: list[Path]) -> pd.DataFrame:
    """Load results from multiple benchmark JSON files into a single DataFrame."""
    frames = []

    for json_file in json_files:
        try:
            format_type = detect_format(json_file)

            if format_type == "google":
                rows = parse_google_benchmark(json_file)
            elif format_type == "pytest":
                rows = parse_pytest_benchmark(json_file)
            else:
                continue

            if rows:
                frames.append(pd.DataFrame(rows))
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def plot_throughput_vs_seq_len(df: pd.DataFrame, output_dir: Path):
    """
    Generate plot for throughput vs. sequence length.
    """
    # Filter relevant benchmarks that have throughput data
    filtered_df = df[
        df["benchmark_name_base"].isin(["BM_PAL_LatencyVsSeqLen", "test_pal_latency_vs_seq_len"])
        & df["throughput_items_per_sec"].notna()
    ]

    if filtered_df.empty:
        print("No throughput data for sequence length plot")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot for each source (cpp, python)
    for source, group in filtered_df.groupby("source"):
        marker = "o" if source == "cpp" else "s"
        label = "Paged Attention (C++)" if source == "cpp" else "Paged Attention (Python)"
        color = "#024645" if source == "cpp" else None  # Dark green for C++
        plt.plot(
            group["seq_len"], group["throughput_items_per_sec"], marker=marker, linestyle="-", label=label, color=color
        )

    # Set plot attributes
    plt.title("Paged Attention: Throughput vs. Sequence Length")
    plt.xlabel("Sequence Length (tokens)")
    plt.ylabel("Throughput (items/second)")
    plt.xscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_dir / "throughput_vs_seq_len.png", dpi=300)
    plt.close()
    print(f"Saved throughput vs. sequence length plot to {output_dir / 'throughput_vs_seq_len.png'}")


def plot_throughput_vs_head_dim(df: pd.DataFrame, output_dir: Path):
    """
    Generate plot for throughput vs. head dimension.
    """
    # Filter relevant benchmarks that have throughput data
    filtered_df = df[
        df["benchmark_name_base"].isin(["BM_PAL_LatencyVsHeadDim", "test_pal_latency_vs_head_dim"])
        & df["throughput_items_per_sec"].notna()
    ]

    if filtered_df.empty:
        print("No throughput data for head dimension plot")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot for each source (cpp, python)
    for source, group in filtered_df.groupby("source"):
        marker = "o" if source == "cpp" else "s"
        label = "Paged Attention (C++)" if source == "cpp" else "Paged Attention (Python)"
        color = "#024645" if source == "cpp" else None  # Dark green for C++
        plt.plot(
            group["head_dim"], group["throughput_items_per_sec"], marker=marker, linestyle="-", label=label, color=color
        )

    # Set plot attributes
    plt.title("Paged Attention: Throughput vs. Head Dimension")
    plt.xlabel("Head Dimension")
    plt.ylabel("Throughput (items/second)")
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_dir / "throughput_vs_head_dim.png", dpi=300)
    plt.close()
    print(f"Saved throughput vs. head dimension plot to {output_dir / 'throughput_vs_head_dim.png'}")


def plot_throughput_vs_num_query_items(df: pd.DataFrame, output_dir: Path):
    """
    Generate plot for throughput vs. number of query items.
    """
    # Filter relevant benchmarks that have throughput data
    filtered_df = df[
        df["benchmark_name_base"].isin(["BM_PAL_LatencyVsNumItems", "test_pal_latency_vs_query_items"])
        & df["throughput_items_per_sec"].notna()
    ]

    if filtered_df.empty:
        print("No throughput data for query items plot")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot for each source (cpp, python)
    for source, group in filtered_df.groupby("source"):
        marker = "o" if source == "cpp" else "s"
        label = "Paged Attention (C++)" if source == "cpp" else "Paged Attention (Python)"
        color = "#024645" if source == "cpp" else None  # Dark green for C++
        plt.plot(
            group["num_query_items"],
            group["throughput_items_per_sec"],
            marker=marker,
            linestyle="-",
            label=label,
            color=color,
        )

    # Set plot attributes
    plt.title("Paged Attention: Throughput vs. Number of Query Items")
    plt.xlabel("Number of Query Items")
    plt.ylabel("Throughput (items/second)")
    plt.xscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_dir / "throughput_vs_num_query_items.png", dpi=300)
    plt.close()
    print(f"Saved throughput vs. query items plot to {output_dir / 'throughput_vs_num_query_items.png'}")


def plot_latency_vs_seq_len(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate consolidated plot for latency vs. sequence length.

    Returns:
        str: The filename of the saved plot
    """
    # Filter relevant benchmarks for PAL and SDPA across all implementations
    pal_cpp_df = df[df["source"] == "cpp_pal"]
    pal_py_df = df[df["source"] == "python_pal"]
    sdpa_cpp_df = df[df["source"] == "cpp_sdpa"]
    sdpa_py_df = df[df["source"] == "python_sdpa"]

    # Filter for sequence length benchmarks
    pal_cpp_df = pal_cpp_df[pal_cpp_df["benchmark_name_base"] == "BM_PAL_LatencyVsSeqLen"]
    pal_py_df = pal_py_df[pal_py_df["benchmark_name_base"] == "test_pal_latency_vs_seq_len"]
    sdpa_cpp_df = sdpa_cpp_df[sdpa_cpp_df["benchmark_name_base"] == "BM_SDPA_LatencyVsSeqLen"]
    sdpa_py_df = sdpa_py_df[sdpa_py_df["benchmark_name_base"] == "test_sdpa_latency_vs_seq_len"]

    # Check if we have any data to plot
    has_data = False
    if not pal_cpp_df.empty or not pal_py_df.empty or not sdpa_cpp_df.empty or not sdpa_py_df.empty:
        has_data = True

    if not has_data:
        print("No data for latency vs. sequence length plot")
        return ""

    # Create the plot with high-quality settings
    plt.figure(figsize=(12, 8), dpi=100)

    # Define consistent styling for implementations
    # Colors
    PAL_CPP_COLOR = "#024645"  # Dark green for PAL C++
    PAL_PY_COLOR = "#026645"  # Slightly different green for PAL Python
    SDPA_CPP_COLOR = "#000000"  # Black for SDPA C++
    SDPA_PY_COLOR = "#000000"  # Black for SDPA Python

    # Line styles
    PAL_CPP_STYLE = "-"  # Solid line for PAL C++
    PAL_PY_STYLE = "--"  # Dashed line for PAL Python
    SDPA_CPP_STYLE = "-."  # Dash-dot line for SDPA C++
    SDPA_PY_STYLE = ":"  # Dotted line for SDPA Python

    # Line widths
    PAL_LINEWIDTH = 2.5  # Bold line for PAL
    SDPA_LINEWIDTH = 1.5  # Standard line for SDPA

    # Markers
    PAL_CPP_MARKER = "o"  # Circle for PAL C++
    PAL_PY_MARKER = "s"  # Square for PAL Python
    SDPA_CPP_MARKER = "^"  # Triangle for SDPA C++
    SDPA_PY_MARKER = "D"  # Diamond for SDPA Python

    # Plot each implementation if data exists
    if not pal_cpp_df.empty:
        plt.plot(
            pal_cpp_df["seq_len"],
            pal_cpp_df["mean_latency_ms"],
            marker=PAL_CPP_MARKER,
            linestyle=PAL_CPP_STYLE,
            linewidth=PAL_LINEWIDTH,
            color=PAL_CPP_COLOR,
            label=r"$\mathbf{Paged\ Attention\ (C++)}$",
        )

    if not pal_py_df.empty:
        plt.plot(
            pal_py_df["seq_len"],
            pal_py_df["mean_latency_ms"],
            marker=PAL_PY_MARKER,
            linestyle=PAL_PY_STYLE,
            linewidth=PAL_LINEWIDTH,
            color=PAL_PY_COLOR,
            label=r"$\mathbf{Paged\ Attention\ (Python)}$",
        )

    if not sdpa_cpp_df.empty:
        plt.plot(
            sdpa_cpp_df["seq_len"],
            sdpa_cpp_df["mean_latency_ms"],
            marker=SDPA_CPP_MARKER,
            linestyle=SDPA_CPP_STYLE,
            linewidth=SDPA_LINEWIDTH,
            color=SDPA_CPP_COLOR,
            label="MLX SDPA (C++)",
        )

    if not sdpa_py_df.empty:
        plt.plot(
            sdpa_py_df["seq_len"],
            sdpa_py_df["mean_latency_ms"],
            marker=SDPA_PY_MARKER,
            linestyle=SDPA_PY_STYLE,
            linewidth=SDPA_LINEWIDTH,
            color=SDPA_PY_COLOR,
            label="MLX SDPA (Python)",
        )

    # Set plot attributes with refined styling
    plt.title("Paged Attention vs. MLX SDPA: Latency vs. Sequence Length", fontsize=16, fontweight="bold")
    plt.xlabel("Sequence Length (tokens)", fontsize=14)
    plt.ylabel("Mean Latency (ms)", fontsize=14)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")

    # Improve tick label readability
    plt.tick_params(axis="both", which="major", labelsize=12)

    # Add reference slope
    # Collect all sequence length data that exists
    all_seq_data = pd.concat([pal_cpp_df, pal_py_df, sdpa_cpp_df, sdpa_py_df])

    if not all_seq_data.empty:
        x_range = all_seq_data["seq_len"].dropna()
        if not x_range.empty:
            x_min, x_max = x_range.min(), x_range.max()
            x_vals = np.linspace(x_min, x_max, 100)

            # Find maximum latency value for scaling
            max_latency = all_seq_data["mean_latency_ms"].max()

            # Linear reference (O(n))
            y_scale = max_latency / x_max if x_max > 0 else 1
            plt.plot(x_vals, y_scale * x_vals, "k--", alpha=0.3, linewidth=1, label="O(n) reference")

    plt.tight_layout()

    # Output filename with consolidated naming
    filename = "latency_vs_seq_len_comparison.png"
    output_path = output_dir / filename

    # Save the plot with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved latency vs. sequence length comparison plot to {output_path}")
    return filename


def plot_latency_vs_head_dim(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate consolidated plot for latency vs. head dimension.

    Returns:
        str: The filename of the saved plot
    """
    # Filter relevant benchmarks for PAL and SDPA across all implementations
    pal_cpp_df = df[df["source"] == "cpp_pal"]
    pal_py_df = df[df["source"] == "python_pal"]
    sdpa_cpp_df = df[df["source"] == "cpp_sdpa"]
    sdpa_py_df = df[df["source"] == "python_sdpa"]

    # Filter for head dimension benchmarks
    pal_cpp_df = pal_cpp_df[pal_cpp_df["benchmark_name_base"] == "BM_PAL_LatencyVsHeadDim"]
    pal_py_df = pal_py_df[pal_py_df["benchmark_name_base"] == "test_pal_latency_vs_head_dim"]
    sdpa_cpp_df = sdpa_cpp_df[sdpa_cpp_df["benchmark_name_base"] == "BM_SDPA_LatencyVsHeadDim"]
    sdpa_py_df = sdpa_py_df[sdpa_py_df["benchmark_name_base"] == "test_sdpa_latency_vs_head_dim"]

    # Check if we have any data to plot
    has_data = False
    if not pal_cpp_df.empty or not pal_py_df.empty or not sdpa_cpp_df.empty or not sdpa_py_df.empty:
        has_data = True

    if not has_data:
        print("No data for latency vs. head dimension plot")
        return ""

    # Create the plot with high-quality settings
    plt.figure(figsize=(12, 8), dpi=100)

    # Define consistent styling for implementations
    # Colors
    PAL_CPP_COLOR = "#024645"  # Dark green for PAL C++
    PAL_PY_COLOR = "#026645"  # Slightly different green for PAL Python
    SDPA_CPP_COLOR = "#000000"  # Black for SDPA C++
    SDPA_PY_COLOR = "#000000"  # Black for SDPA Python

    # Line styles
    PAL_CPP_STYLE = "-"  # Solid line for PAL C++
    PAL_PY_STYLE = "--"  # Dashed line for PAL Python
    SDPA_CPP_STYLE = "-."  # Dash-dot line for SDPA C++
    SDPA_PY_STYLE = ":"  # Dotted line for SDPA Python

    # Line widths
    PAL_LINEWIDTH = 2.5  # Bold line for PAL
    SDPA_LINEWIDTH = 1.5  # Standard line for SDPA

    # Markers
    PAL_CPP_MARKER = "o"  # Circle for PAL C++
    PAL_PY_MARKER = "s"  # Square for PAL Python
    SDPA_CPP_MARKER = "^"  # Triangle for SDPA C++
    SDPA_PY_MARKER = "D"  # Diamond for SDPA Python

    # Plot each implementation if data exists
    if not pal_cpp_df.empty:
        plt.plot(
            pal_cpp_df["head_dim"],
            pal_cpp_df["mean_latency_ms"],
            marker=PAL_CPP_MARKER,
            linestyle=PAL_CPP_STYLE,
            linewidth=PAL_LINEWIDTH,
            color=PAL_CPP_COLOR,
            label=r"$\mathbf{Paged\ Attention\ (C++)}$",
        )

    if not pal_py_df.empty:
        plt.plot(
            pal_py_df["head_dim"],
            pal_py_df["mean_latency_ms"],
            marker=PAL_PY_MARKER,
            linestyle=PAL_PY_STYLE,
            linewidth=PAL_LINEWIDTH,
            color=PAL_PY_COLOR,
            label=r"$\mathbf{Paged\ Attention\ (Python)}$",
        )

    if not sdpa_cpp_df.empty:
        plt.plot(
            sdpa_cpp_df["head_dim"],
            sdpa_cpp_df["mean_latency_ms"],
            marker=SDPA_CPP_MARKER,
            linestyle=SDPA_CPP_STYLE,
            linewidth=SDPA_LINEWIDTH,
            color=SDPA_CPP_COLOR,
            label="MLX SDPA (C++)",
        )

    if not sdpa_py_df.empty:
        plt.plot(
            sdpa_py_df["head_dim"],
            sdpa_py_df["mean_latency_ms"],
            marker=SDPA_PY_MARKER,
            linestyle=SDPA_PY_STYLE,
            linewidth=SDPA_LINEWIDTH,
            color=SDPA_PY_COLOR,
            label="MLX SDPA (Python)",
        )

    # Set plot attributes with refined styling
    plt.title("Paged Attention vs. MLX SDPA: Latency vs. Head Dimension", fontsize=16, fontweight="bold")
    plt.xlabel("Head Dimension", fontsize=14)
    plt.ylabel("Mean Latency (ms)", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")

    # Improve tick label readability
    plt.tick_params(axis="both", which="major", labelsize=12)

    # Add reference O(d²) curve
    all_dim_data = pd.concat([pal_cpp_df, pal_py_df, sdpa_cpp_df, sdpa_py_df])

    if not all_dim_data.empty:
        x_range = all_dim_data["head_dim"].dropna()
        if not x_range.empty:
            x_min, x_max = x_range.min(), x_range.max()
            x_vals = np.linspace(x_min, x_max, 100)

            # Calculate scaling factor
            if x_max > 0:
                max_latency = all_dim_data["mean_latency_ms"].max()
                y_scale = max_latency / (x_max**2)
                plt.plot(x_vals, y_scale * x_vals**2, "k--", alpha=0.3, linewidth=1, label="O(d²) reference")

    plt.tight_layout()

    # Output filename with consolidated naming
    filename = "latency_vs_head_dim_comparison.png"
    output_path = output_dir / filename

    # Save the plot with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved latency vs. head dimension comparison plot to {output_path}")
    return filename


def plot_latency_vs_num_query_items(df: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate consolidated plot for latency vs. number of query items / batch size.

    This plot uses the 'effective_items' metric for comparison across all implementations.

    Returns:
        str: The filename of the saved plot
    """
    # Filter relevant benchmarks for PAL and SDPA across all implementations
    pal_cpp_df = df[df["source"] == "cpp_pal"]
    pal_py_df = df[df["source"] == "python_pal"]
    sdpa_cpp_df = df[df["source"] == "cpp_sdpa"]
    sdpa_py_df = df[df["source"] == "python_sdpa"]

    # Filter for query items / batch size benchmarks
    pal_cpp_df = pal_cpp_df[pal_cpp_df["benchmark_name_base"] == "BM_PAL_LatencyVsNumItems"]
    pal_py_df = pal_py_df[pal_py_df["benchmark_name_base"] == "test_pal_latency_vs_query_items"]
    sdpa_cpp_df = sdpa_cpp_df[sdpa_cpp_df["benchmark_name_base"] == "BM_SDPA_LatencyVsNumItems"]
    sdpa_py_df = sdpa_py_df[sdpa_py_df["benchmark_name_base"] == "test_sdpa_latency_vs_batch_size"]

    # Check if we have any data to plot
    has_data = False
    if not pal_cpp_df.empty or not pal_py_df.empty or not sdpa_cpp_df.empty or not sdpa_py_df.empty:
        has_data = True

    if not has_data:
        print("No data for latency vs. query items / batch size plot")
        return ""

    # Create the plot with high-quality settings
    plt.figure(figsize=(12, 8), dpi=100)

    # Define consistent styling for implementations
    # Colors
    PAL_CPP_COLOR = "#024645"  # Dark green for PAL C++
    PAL_PY_COLOR = "#026645"  # Slightly different green for PAL Python
    SDPA_CPP_COLOR = "#000000"  # Black for SDPA C++
    SDPA_PY_COLOR = "#000000"  # Black for SDPA Python

    # Line styles
    PAL_CPP_STYLE = "-"  # Solid line for PAL C++
    PAL_PY_STYLE = "--"  # Dashed line for PAL Python
    SDPA_CPP_STYLE = "-."  # Dash-dot line for SDPA C++
    SDPA_PY_STYLE = ":"  # Dotted line for SDPA Python

    # Line widths
    PAL_LINEWIDTH = 2.5  # Bold line for PAL
    SDPA_LINEWIDTH = 1.5  # Standard line for SDPA

    # Markers
    PAL_CPP_MARKER = "o"  # Circle for PAL C++
    PAL_PY_MARKER = "s"  # Square for PAL Python
    SDPA_CPP_MARKER = "^"  # Triangle for SDPA C++
    SDPA_PY_MARKER = "D"  # Diamond for SDPA Python

    # Plot each implementation if data exists
    if not pal_cpp_df.empty:
        plt.plot(
            pal_cpp_df["effective_items"],
            pal_cpp_df["mean_latency_ms"],
            marker=PAL_CPP_MARKER,
            linestyle=PAL_CPP_STYLE,
            linewidth=PAL_LINEWIDTH,
            color=PAL_CPP_COLOR,
            label=r"$\mathbf{Paged\ Attention\ (C++)}$",
        )

    if not pal_py_df.empty:
        plt.plot(
            pal_py_df["effective_items"],
            pal_py_df["mean_latency_ms"],
            marker=PAL_PY_MARKER,
            linestyle=PAL_PY_STYLE,
            linewidth=PAL_LINEWIDTH,
            color=PAL_PY_COLOR,
            label=r"$\mathbf{Paged\ Attention\ (Python)}$",
        )

    if not sdpa_cpp_df.empty:
        plt.plot(
            sdpa_cpp_df["effective_items"],
            sdpa_cpp_df["mean_latency_ms"],
            marker=SDPA_CPP_MARKER,
            linestyle=SDPA_CPP_STYLE,
            linewidth=SDPA_LINEWIDTH,
            color=SDPA_CPP_COLOR,
            label="MLX SDPA (C++)",
        )

    if not sdpa_py_df.empty:
        plt.plot(
            sdpa_py_df["effective_items"],
            sdpa_py_df["mean_latency_ms"],
            marker=SDPA_PY_MARKER,
            linestyle=SDPA_PY_STYLE,
            linewidth=SDPA_LINEWIDTH,
            color=SDPA_PY_COLOR,
            label="MLX SDPA (Python)",
        )

    # Set plot attributes with refined styling
    plt.title("Paged Attention vs. MLX SDPA: Latency vs. Processing Items", fontsize=16, fontweight="bold")
    plt.xlabel("Number of Effective Items (Batch Size x Heads)", fontsize=14)
    plt.ylabel("Mean Latency (ms)", fontsize=14)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")

    # Improve tick label readability
    plt.tick_params(axis="both", which="major", labelsize=12)

    # Add reference O(n) line for linear scaling
    all_item_data = pd.concat([pal_cpp_df, pal_py_df, sdpa_cpp_df, sdpa_py_df])

    if not all_item_data.empty:
        x_range = all_item_data["effective_items"].dropna()
        if not x_range.empty:
            x_min, x_max = x_range.min(), x_range.max()
            x_vals = np.linspace(x_min, x_max, 100)

            # Calculate scaling factor
            if x_max > 0:
                max_latency = all_item_data["mean_latency_ms"].max()
                y_scale = max_latency / x_max
                plt.plot(x_vals, y_scale * x_vals, "k--", alpha=0.3, linewidth=1, label="O(n) reference")

    plt.tight_layout()

    # Output filename with consolidated naming
    filename = "latency_vs_num_items_comparison.png"
    output_path = output_dir / filename

    # Save the plot with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved latency vs. query items comparison plot to {output_path}")
    return filename


def plot_model_configs_latency(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """
    Generate plot for latency of different model configurations.

    Returns:
        dict[str, str]: Dictionary mapping plot descriptions to filenames
    """
    # Dictionary to track all plot filenames created
    plot_filenames = {}

    # Gather all model config benchmarks across implementations
    pal_py_df = df[(df["source"] == "python_pal") & (df["benchmark_name_base"] == "test_pal_latency_model_configs")]
    pal_cpp_df = df[(df["source"] == "cpp_pal") & (df["benchmark_name_base"].str.contains("ModelConfig"))]
    sdpa_py_df = df[(df["source"] == "python_sdpa") & (df["benchmark_name_base"] == "test_sdpa_latency_model_configs")]
    sdpa_cpp_df = df[(df["source"] == "cpp_sdpa") & (df["benchmark_name_base"].str.contains("ModelConfig"))]

    # Create a consolidated comparison plot across all available implementations
    model_configs_to_plot = set()

    if not pal_py_df.empty:
        model_configs_to_plot.update(pal_py_df["model_config_name"].dropna())
    if not pal_cpp_df.empty:
        model_configs_to_plot.update(pal_cpp_df["model_config_name"].dropna())
    if not sdpa_py_df.empty:
        model_configs_to_plot.update(sdpa_py_df["model_config_name"].dropna())
    if not sdpa_cpp_df.empty:
        model_configs_to_plot.update(sdpa_cpp_df["model_config_name"].dropna())

    # Skip if no config data available
    if not model_configs_to_plot:
        print("No model configuration benchmark data available")
        return plot_filenames

    # Prepare data for plot - we'll create a wide-format DataFrame
    plot_data = []

    for config in sorted(model_configs_to_plot):
        row_data = {"model_config_name": config}

        # Add PAL Python latency if available
        if not pal_py_df.empty:
            config_data = pal_py_df[pal_py_df["model_config_name"] == config]
            if not config_data.empty:
                row_data["pal_py_latency_ms"] = config_data["mean_latency_ms"].iloc[0]

        # Add PAL C++ latency if available
        if not pal_cpp_df.empty:
            config_data = pal_cpp_df[pal_cpp_df["model_config_name"] == config]
            if not config_data.empty:
                row_data["pal_cpp_latency_ms"] = config_data["mean_latency_ms"].iloc[0]

        # Add SDPA Python latency if available
        if not sdpa_py_df.empty:
            config_data = sdpa_py_df[sdpa_py_df["model_config_name"] == config]
            if not config_data.empty:
                row_data["sdpa_py_latency_ms"] = config_data["mean_latency_ms"].iloc[0]

        # Add SDPA C++ latency if available
        if not sdpa_cpp_df.empty:
            config_data = sdpa_cpp_df[sdpa_cpp_df["model_config_name"] == config]
            if not config_data.empty:
                row_data["sdpa_cpp_latency_ms"] = config_data["mean_latency_ms"].iloc[0]

        # Add speedup ratios where possible
        # PAL Python / SDPA Python
        if "pal_py_latency_ms" in row_data and "sdpa_py_latency_ms" in row_data and row_data["sdpa_py_latency_ms"] > 0:
            row_data["pal_py_sdpa_py_ratio"] = row_data["pal_py_latency_ms"] / row_data["sdpa_py_latency_ms"]

        # PAL C++ / SDPA C++
        if (
            "pal_cpp_latency_ms" in row_data
            and "sdpa_cpp_latency_ms" in row_data
            and row_data["sdpa_cpp_latency_ms"] > 0
        ):
            row_data["pal_cpp_sdpa_cpp_ratio"] = row_data["pal_cpp_latency_ms"] / row_data["sdpa_cpp_latency_ms"]

        # Only add rows that have at least one latency value
        if any(key.endswith("latency_ms") for key in row_data):
            plot_data.append(row_data)

    if not plot_data:
        print("No model configuration latency data available after filtering")
        return plot_filenames

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)

    # Sort by model config name for consistent ordering
    plot_df = plot_df.sort_values("model_config_name")

    # Figure out which columns are available for plotting
    available_columns = [
        col
        for col in ["pal_py_latency_ms", "pal_cpp_latency_ms", "sdpa_py_latency_ms", "sdpa_cpp_latency_ms"]
        if col in plot_df.columns
    ]

    if available_columns:
        # Create the grouped bar chart with high-quality settings
        plt.figure(figsize=(14, 8), dpi=100)

        # Define colors that match our line plot scheme
        colors = {
            "pal_cpp_latency_ms": "#024645",  # Dark green for PAL C++
            "pal_py_latency_ms": "#026645",  # Slightly different green for PAL Python
            "sdpa_cpp_latency_ms": "#000000",  # Black for SDPA C++
            "sdpa_py_latency_ms": "#444444",  # Dark gray for SDPA Python for better differentiation
        }

        # Define labels
        labels = {
            "pal_cpp_latency_ms": r"$\mathbf{Paged\ Attention\ (C++)}$",
            "pal_py_latency_ms": r"$\mathbf{Paged\ Attention\ (Python)}$",
            "sdpa_cpp_latency_ms": "MLX SDPA (C++)",
            "sdpa_py_latency_ms": "MLX SDPA (Python)",
        }

        # Define bar width and positions
        bar_width = 0.8 / len(available_columns)
        positions = np.arange(len(plot_df))

        # Keep track of bars for annotating speedup ratios
        all_bars = {}

        # Plot each implementation's bars
        for i, col in enumerate(available_columns):
            # Calculate offset for this set of bars
            offset = (i - len(available_columns) / 2 + 0.5) * bar_width

            # Create bars for this implementation
            bars = plt.bar(
                positions + offset,
                plot_df[col].fillna(0),
                width=bar_width,
                label=labels.get(col, col),
                color=colors.get(col),
                edgecolor="black",
                linewidth=0.5,
            )

            # Store bars for later adding speedup ratios
            all_bars[col] = bars

            # Add latency values on top of bars
            for j, bar in enumerate(bars):
                if pd.notna(plot_df[col].iloc[j]) and plot_df[col].iloc[j] > 0:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{plot_df[col].iloc[j]:.2f}",
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=8,
                    )

        # Add speedup ratios for Python implementations
        if "pal_py_sdpa_py_ratio" in plot_df.columns:
            for i, (ratio, is_valid) in enumerate(
                zip(plot_df["pal_py_sdpa_py_ratio"], plot_df["pal_py_sdpa_py_ratio"].notna(), strict=True)
            ):
                if is_valid:
                    # Find the maximum bar height at this position
                    max_height = max(bar[i].get_height() for bars in all_bars.values() for bar in bars if i < len(bar))
                    plt.text(
                        positions[i],
                        max_height + 1.5,
                        f"P-Ratio: {ratio:.2f}x",
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=9,
                        color="black",
                    )

        # Add speedup ratios for C++ implementations
        if "pal_cpp_sdpa_cpp_ratio" in plot_df.columns:
            for i, (ratio, is_valid) in enumerate(
                zip(plot_df["pal_cpp_sdpa_cpp_ratio"], plot_df["pal_cpp_sdpa_cpp_ratio"].notna(), strict=True)
            ):
                if is_valid:
                    # Find the maximum bar height at this position
                    max_height = max(bar[i].get_height() for bars in all_bars.values() for bar in bars if i < len(bar))
                    plt.text(
                        positions[i],
                        max_height + 3.0,
                        f"C-Ratio: {ratio:.2f}x",
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=9,
                        color="black",
                    )

        # Set x-axis tick labels
        plt.xticks(positions, plot_df["model_config_name"].tolist(), rotation=45, ha="right", fontsize=10)

        # Set plot attributes with refined styling
        plt.title("Paged Attention vs. MLX SDPA: Model Configuration Latencies", fontsize=16, fontweight="bold")
        plt.xlabel("Model Configuration", fontsize=14)
        plt.ylabel("Mean Latency (ms)", fontsize=14)
        plt.grid(True, axis="y", ls="-", alpha=0.2, color="lightgray")

        # Improve tick label readability
        plt.tick_params(axis="both", which="major", labelsize=12)

        # Add some padding at the top for the ratios
        plt.ylim(0, plt.ylim()[1] * 1.2)

        plt.tight_layout()

        # Save the consolidated plot
        filename = "model_configs_latency_comparison.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved model configurations latency comparison plot to {output_path}")
        plot_filenames["model_configs"] = filename

    return plot_filenames


def generate_json_report(df: pd.DataFrame, output_dir: Path, plot_filenames: dict) -> None:
    """
    Generate a structured JSON report containing benchmark data and plot filenames.

    Args:
        df: DataFrame containing processed benchmark results
        output_dir: Directory to save the JSON report
        plot_filenames: Dictionary mapping plot types to filenames
    """
    import datetime

    # Create the report structure
    report = {
        "summary_metrics": {},
        "plot_files": plot_filenames,
        "generation_timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Process latency vs sequence length data
    seq_len_data = []

    # PAL C++ data
    pal_cpp_seq_df = df[(df["source"] == "cpp_pal") & (df["benchmark_name_base"] == "BM_PAL_LatencyVsSeqLen")]
    if not pal_cpp_seq_df.empty:
        for _, row in pal_cpp_seq_df.iterrows():
            seq_len_data.append(
                {
                    "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                    "source": "cpp_pal",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # PAL Python data
    pal_py_seq_df = df[(df["source"] == "python_pal") & (df["benchmark_name_base"] == "test_pal_latency_vs_seq_len")]
    if not pal_py_seq_df.empty:
        for _, row in pal_py_seq_df.iterrows():
            seq_len_data.append(
                {
                    "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                    "source": "python_pal",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # SDPA C++ data
    sdpa_cpp_seq_df = df[(df["source"] == "cpp_sdpa") & (df["benchmark_name_base"] == "BM_SDPA_LatencyVsSeqLen")]
    if not sdpa_cpp_seq_df.empty:
        for _, row in sdpa_cpp_seq_df.iterrows():
            seq_len_data.append(
                {
                    "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                    "source": "cpp_sdpa",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # SDPA Python data
    sdpa_py_seq_df = df[(df["source"] == "python_sdpa") & (df["benchmark_name_base"] == "test_sdpa_latency_vs_seq_len")]
    if not sdpa_py_seq_df.empty:
        for _, row in sdpa_py_seq_df.iterrows():
            seq_len_data.append(
                {
                    "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                    "source": "python_sdpa",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    if seq_len_data:
        report["summary_metrics"]["latency_vs_seq_len"] = seq_len_data

    # Process latency vs head dimension data
    head_dim_data = []

    # PAL C++ data
    pal_cpp_head_df = df[(df["source"] == "cpp_pal") & (df["benchmark_name_base"] == "BM_PAL_LatencyVsHeadDim")]
    if not pal_cpp_head_df.empty:
        for _, row in pal_cpp_head_df.iterrows():
            head_dim_data.append(
                {
                    "head_dim": int(row["head_dim"]) if pd.notna(row["head_dim"]) else None,
                    "source": "cpp_pal",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # PAL Python data
    pal_py_head_df = df[(df["source"] == "python_pal") & (df["benchmark_name_base"] == "test_pal_latency_vs_head_dim")]
    if not pal_py_head_df.empty:
        for _, row in pal_py_head_df.iterrows():
            head_dim_data.append(
                {
                    "head_dim": int(row["head_dim"]) if pd.notna(row["head_dim"]) else None,
                    "source": "python_pal",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # SDPA C++ data
    sdpa_cpp_head_df = df[(df["source"] == "cpp_sdpa") & (df["benchmark_name_base"] == "BM_SDPA_LatencyVsHeadDim")]
    if not sdpa_cpp_head_df.empty:
        for _, row in sdpa_cpp_head_df.iterrows():
            head_dim_data.append(
                {
                    "head_dim": int(row["head_dim"]) if pd.notna(row["head_dim"]) else None,
                    "source": "cpp_sdpa",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # SDPA Python data
    sdpa_py_head_df = df[
        (df["source"] == "python_sdpa") & (df["benchmark_name_base"] == "test_sdpa_latency_vs_head_dim")
    ]
    if not sdpa_py_head_df.empty:
        for _, row in sdpa_py_head_df.iterrows():
            head_dim_data.append(
                {
                    "head_dim": int(row["head_dim"]) if pd.notna(row["head_dim"]) else None,
                    "source": "python_sdpa",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    if head_dim_data:
        report["summary_metrics"]["latency_vs_head_dim"] = head_dim_data

    # Process latency vs number of query items / batch size data
    num_items_data = []

    # PAL C++ data
    pal_cpp_items_df = df[(df["source"] == "cpp_pal") & (df["benchmark_name_base"] == "BM_PAL_LatencyVsNumItems")]
    if not pal_cpp_items_df.empty:
        for _, row in pal_cpp_items_df.iterrows():
            num_items_data.append(
                {
                    "effective_items": int(row["effective_items"]) if pd.notna(row["effective_items"]) else None,
                    "num_query_items": int(row["num_query_items"]) if pd.notna(row["num_query_items"]) else None,
                    "source": "cpp_pal",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # PAL Python data
    pal_py_items_df = df[
        (df["source"] == "python_pal") & (df["benchmark_name_base"] == "test_pal_latency_vs_query_items")
    ]
    if not pal_py_items_df.empty:
        for _, row in pal_py_items_df.iterrows():
            num_items_data.append(
                {
                    "effective_items": int(row["effective_items"]) if pd.notna(row["effective_items"]) else None,
                    "num_query_items": int(row["num_query_items"]) if pd.notna(row["num_query_items"]) else None,
                    "source": "python_pal",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # SDPA C++ data
    sdpa_cpp_items_df = df[(df["source"] == "cpp_sdpa") & (df["benchmark_name_base"] == "BM_SDPA_LatencyVsNumItems")]
    if not sdpa_cpp_items_df.empty:
        for _, row in sdpa_cpp_items_df.iterrows():
            num_items_data.append(
                {
                    "effective_items": int(row["effective_items"]) if pd.notna(row["effective_items"]) else None,
                    "batch_size": int(row["batch_size"]) if pd.notna(row["batch_size"]) else None,
                    "source": "cpp_sdpa",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    # SDPA Python data
    sdpa_py_items_df = df[
        (df["source"] == "python_sdpa") & (df["benchmark_name_base"] == "test_sdpa_latency_vs_batch_size")
    ]
    if not sdpa_py_items_df.empty:
        for _, row in sdpa_py_items_df.iterrows():
            num_items_data.append(
                {
                    "effective_items": int(row["effective_items"]) if pd.notna(row["effective_items"]) else None,
                    "batch_size": int(row["batch_size"]) if pd.notna(row["batch_size"]) else None,
                    "source": "python_sdpa",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "throughput_items_per_sec": float(row["throughput_items_per_sec"])
                    if pd.notna(row["throughput_items_per_sec"])
                    else None,
                }
            )

    if num_items_data:
        report["summary_metrics"]["latency_vs_num_items"] = num_items_data

    # Process model configurations data
    model_config_data = []

    # PAL C++ data
    pal_cpp_model_df = df[(df["source"] == "cpp_pal") & (df["benchmark_name_base"].str.contains("ModelConfig"))]
    if not pal_cpp_model_df.empty:
        for _, row in pal_cpp_model_df.iterrows():
            model_config_data.append(
                {
                    "model_config_name": row["model_config_name"] if pd.notna(row["model_config_name"]) else None,
                    "source": "cpp_pal",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "num_q_heads": int(row["num_q_heads"]) if pd.notna(row["num_q_heads"]) else None,
                    "num_kv_heads": int(row["num_kv_heads"]) if pd.notna(row["num_kv_heads"]) else None,
                    "head_dim": int(row["head_dim"]) if pd.notna(row["head_dim"]) else None,
                    "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                }
            )

    # PAL Python data
    pal_py_model_df = df[
        (df["source"] == "python_pal") & (df["benchmark_name_base"] == "test_pal_latency_model_configs")
    ]
    if not pal_py_model_df.empty:
        for _, row in pal_py_model_df.iterrows():
            model_config_data.append(
                {
                    "model_config_name": row["model_config_name"] if pd.notna(row["model_config_name"]) else None,
                    "source": "python_pal",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "num_q_heads": int(row["num_q_heads"]) if pd.notna(row["num_q_heads"]) else None,
                    "num_kv_heads": int(row["num_kv_heads"]) if pd.notna(row["num_kv_heads"]) else None,
                    "head_dim": int(row["head_dim"]) if pd.notna(row["head_dim"]) else None,
                    "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                }
            )

    # SDPA C++ data
    sdpa_cpp_model_df = df[(df["source"] == "cpp_sdpa") & (df["benchmark_name_base"].str.contains("ModelConfig"))]
    if not sdpa_cpp_model_df.empty:
        for _, row in sdpa_cpp_model_df.iterrows():
            model_config_data.append(
                {
                    "model_config_name": row["model_config_name"] if pd.notna(row["model_config_name"]) else None,
                    "source": "cpp_sdpa",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "batch_size": int(row["batch_size"]) if pd.notna(row["batch_size"]) else None,
                    "num_q_heads": int(row["num_q_heads"]) if pd.notna(row["num_q_heads"]) else None,
                    "num_kv_heads": int(row["num_kv_heads"]) if pd.notna(row["num_kv_heads"]) else None,
                    "head_dim": int(row["head_dim"]) if pd.notna(row["head_dim"]) else None,
                    "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                }
            )

    # SDPA Python data
    sdpa_py_model_df = df[
        (df["source"] == "python_sdpa") & (df["benchmark_name_base"] == "test_sdpa_latency_model_configs")
    ]
    if not sdpa_py_model_df.empty:
        for _, row in sdpa_py_model_df.iterrows():
            model_config_data.append(
                {
                    "model_config_name": row["model_config_name"] if pd.notna(row["model_config_name"]) else None,
                    "source": "python_sdpa",
                    "mean_latency_ms": float(row["mean_latency_ms"]) if pd.notna(row["mean_latency_ms"]) else None,
                    "batch_size": int(row["batch_size"]) if pd.notna(row["batch_size"]) else None,
                    "num_q_heads": int(row["num_q_heads"]) if pd.notna(row["num_q_heads"]) else None,
                    "num_kv_heads": int(row["num_kv_heads"]) if pd.notna(row["num_kv_heads"]) else None,
                    "head_dim": int(row["head_dim"]) if pd.notna(row["head_dim"]) else None,
                    "seq_len": int(row["seq_len"]) if pd.notna(row["seq_len"]) else None,
                }
            )

    if model_config_data:
        report["summary_metrics"]["model_configs"] = model_config_data

    # Save the JSON report
    with open(output_dir / "results.json", "w") as f:
        import json

        json.dump(report, f, indent=2)

    print(f"Saved benchmark results JSON to {output_dir / 'results.json'}")


def extract_and_normalize_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and normalize parameters from benchmark names and raw params.
    """
    # Initialize new parameter columns
    df["seq_len"] = pd.NA
    df["head_dim"] = pd.NA
    df["num_query_items"] = pd.NA
    df["num_q_heads"] = pd.NA
    df["num_kv_heads"] = pd.NA
    df["model_config_name"] = pd.NA
    df["batch_size"] = pd.NA  # Add batch_size column for SDPA benchmarks
    df["effective_items"] = pd.NA  # Common metric for comparing PAL and SDPA

    # Default values for parameters not directly specified in benchmark names
    DEFAULT_HEAD_DIM = 128
    DEFAULT_NUM_Q_HEADS = 1
    DEFAULT_NUM_KV_HEADS = 1
    DEFAULT_SEQ_LEN = 128
    DEFAULT_NUM_QUERY_ITEMS = 64
    DEFAULT_TOKENS_PER_PAGE = 64
    DEFAULT_NUM_SEQUENCES_IN_BATCH = 1
    DEFAULT_BATCH_SIZE = 64  # Default batch size for SDPA

    # Add tokens_per_page and num_sequences_in_batch columns
    df["tokens_per_page"] = pd.NA
    df["num_sequences_in_batch"] = pd.NA

    # Process each row
    for idx, row in df.iterrows():
        base_name = row["benchmark_name_base"]
        params_str = row["params_str"]
        source = row["source"]

        # Handle C++ PAL benchmarks
        if source == "cpp_pal":
            if base_name == "BM_PAL_LatencyVsSeqLen":
                try:
                    # Format: BM_PAL_LatencyVsSeqLen/64/128 (num_query_items/seq_len)
                    # Strip any _mean, _median, etc. suffixes from parameter parts
                    parts = []
                    for part in params_str.split("/"):
                        # Remove any suffix like _mean, _median, etc.
                        clean_part = re.sub(r"_[a-z]+$", "", part)
                        parts.append(clean_part)

                    if len(parts) >= 2:
                        # First part is num_query_items (possibly with multiplication)
                        if "*" in parts[0]:
                            num_items_parts = parts[0].split("*")
                            if len(num_items_parts) == 2:
                                df.at[idx, "num_query_items"] = int(num_items_parts[0])
                                df.at[idx, "num_q_heads"] = int(num_items_parts[1])
                        else:
                            df.at[idx, "num_query_items"] = int(parts[0])
                            df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS

                        # Second part is seq_len
                        df.at[idx, "seq_len"] = int(parts[1])

                        # Set defaults for other parameters
                        df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                        df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                        df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                        df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                        # Calculate effective_items for comparison
                        df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing BM_PAL_LatencyVsSeqLen parameters for row {idx}: {e}")
                    print(f"  params_str: '{params_str}'")

            elif base_name == "BM_PAL_LatencyVsHeadDim":
                try:
                    # Format: BM_PAL_LatencyVsHeadDim/64/128 (num_query_items/head_dim)
                    # Strip any _mean, _median, etc. suffixes from parameter parts
                    parts = []
                    for part in params_str.split("/"):
                        # Remove any suffix like _mean, _median, etc.
                        clean_part = re.sub(r"_[a-z]+$", "", part)
                        parts.append(clean_part)

                    if len(parts) >= 2:
                        # First part is num_query_items
                        if "*" in parts[0]:
                            num_items_parts = parts[0].split("*")
                            if len(num_items_parts) == 2:
                                df.at[idx, "num_query_items"] = int(num_items_parts[0])
                                df.at[idx, "num_q_heads"] = int(num_items_parts[1])
                        else:
                            df.at[idx, "num_query_items"] = int(parts[0])
                            df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS

                        # Second part is head_dim
                        df.at[idx, "head_dim"] = int(parts[1])

                        # Set defaults for other parameters
                        df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                        df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                        df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                        df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                        # Calculate effective_items for comparison
                        df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing BM_PAL_LatencyVsHeadDim parameters for row {idx}: {e}")
                    print(f"  params_str: '{params_str}'")

            elif base_name == "BM_PAL_LatencyVsNumItems":
                try:
                    # Format: BM_PAL_LatencyVsNumItems/64 (num_query_items)
                    # Remove any suffix like _mean, _median, etc.
                    clean_param = re.sub(r"_[a-z]+$", "", params_str)

                    if clean_param:
                        df.at[idx, "num_query_items"] = int(clean_param)

                        # Set defaults for other parameters
                        df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                        df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                        df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                        df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                        df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                        df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                        # Calculate effective_items for comparison
                        df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing BM_PAL_LatencyVsNumItems parameters for row {idx}: {e}")
                    print(f"  params_str: '{params_str}'")

            # Add handling for PAL model config benchmarks if they exist in C++
            elif base_name.startswith("BM_PAL_ModelConfig_"):
                # Extract model name from benchmark name
                model_name = base_name.replace("BM_PAL_ModelConfig_", "")
                df.at[idx, "model_config_name"] = model_name

                # Set defaults based on model config - adjust these as needed
                if "Llama3_70B" in model_name:
                    df.at[idx, "num_query_items"] = 64 * 64
                    df.at[idx, "num_q_heads"] = 64
                    df.at[idx, "num_kv_heads"] = 8
                    df.at[idx, "head_dim"] = 128
                    df.at[idx, "seq_len"] = 1024
                elif "Qwen_8B" in model_name:
                    df.at[idx, "num_query_items"] = 64 * 32
                    df.at[idx, "num_q_heads"] = 32
                    df.at[idx, "num_kv_heads"] = 32
                    df.at[idx, "head_dim"] = 128
                    df.at[idx, "seq_len"] = 1024
                elif "Qwen2_5_72B" in model_name:
                    df.at[idx, "num_query_items"] = 64 * 128
                    df.at[idx, "num_q_heads"] = 128
                    df.at[idx, "num_kv_heads"] = 8
                    df.at[idx, "head_dim"] = 128
                    df.at[idx, "seq_len"] = 1024

                # Calculate effective_items for comparison
                df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

        # Handle C++ SDPA benchmarks
        elif source == "cpp_sdpa":
            if base_name == "BM_SDPA_LatencyVsSeqLen":
                try:
                    # Format: BM_SDPA_LatencyVsSeqLen/64/128 (batch_size/seq_len)
                    parts = []
                    for part in params_str.split("/"):
                        # Remove any suffix like _mean, _median, etc.
                        clean_part = re.sub(r"_[a-z]+$", "", part)
                        parts.append(clean_part)

                    if len(parts) >= 2:
                        # First part is batch_size
                        df.at[idx, "batch_size"] = int(parts[0])

                        # Second part is seq_len
                        df.at[idx, "seq_len"] = int(parts[1])

                        # Set defaults for other parameters
                        df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                        df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                        df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS

                        # Calculate equivalent num_query_items for comparison with PAL
                        df.at[idx, "num_query_items"] = df.at[idx, "batch_size"] * DEFAULT_NUM_Q_HEADS

                        # Calculate effective_items for comparison
                        df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing BM_SDPA_LatencyVsSeqLen parameters for row {idx}: {e}")
                    print(f"  params_str: '{params_str}'")

            elif base_name == "BM_SDPA_LatencyVsHeadDim":
                try:
                    # Format: BM_SDPA_LatencyVsHeadDim/64/128 (batch_size/head_dim)
                    parts = []
                    for part in params_str.split("/"):
                        # Remove any suffix like _mean, _median, etc.
                        clean_part = re.sub(r"_[a-z]+$", "", part)
                        parts.append(clean_part)

                    if len(parts) >= 2:
                        # First part is batch_size
                        df.at[idx, "batch_size"] = int(parts[0])

                        # Second part is head_dim
                        df.at[idx, "head_dim"] = int(parts[1])

                        # Set defaults for other parameters
                        df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                        df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                        df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS

                        # Calculate equivalent num_query_items for comparison with PAL
                        df.at[idx, "num_query_items"] = df.at[idx, "batch_size"] * DEFAULT_NUM_Q_HEADS

                        # Calculate effective_items for comparison
                        df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing BM_SDPA_LatencyVsHeadDim parameters for row {idx}: {e}")
                    print(f"  params_str: '{params_str}'")

            elif base_name == "BM_SDPA_LatencyVsNumItems":
                try:
                    # Format: BM_SDPA_LatencyVsNumItems/64 (batch_size)
                    # Remove any suffix like _mean, _median, etc.
                    clean_param = re.sub(r"_[a-z]+$", "", params_str)

                    if clean_param:
                        df.at[idx, "batch_size"] = int(clean_param)

                        # Set defaults for other parameters
                        df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                        df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                        df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                        df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS

                        # Calculate equivalent num_query_items for comparison with PAL
                        df.at[idx, "num_query_items"] = df.at[idx, "batch_size"] * DEFAULT_NUM_Q_HEADS

                        # Calculate effective_items for comparison
                        df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                except (ValueError, IndexError) as e:
                    print(f"Error parsing BM_SDPA_LatencyVsNumItems parameters for row {idx}: {e}")
                    print(f"  params_str: '{params_str}'")

            # Add handling for SDPA model config benchmarks
            elif base_name.startswith("BM_SDPA_ModelConfig_"):
                # Extract model name from benchmark name
                model_name = base_name.replace("BM_SDPA_ModelConfig_", "")
                df.at[idx, "model_config_name"] = model_name

                # Set defaults based on model config - adjust based on actual C++ implementation
                if "Llama3_70B" in model_name:
                    df.at[idx, "batch_size"] = 4
                    df.at[idx, "num_q_heads"] = 64
                    df.at[idx, "num_kv_heads"] = 8
                    df.at[idx, "head_dim"] = 128
                    df.at[idx, "seq_len"] = 1024
                elif "Qwen_8B" in model_name:
                    df.at[idx, "batch_size"] = 4
                    df.at[idx, "num_q_heads"] = 32
                    df.at[idx, "num_kv_heads"] = 32
                    df.at[idx, "head_dim"] = 128
                    df.at[idx, "seq_len"] = 1024
                elif "Qwen2_5_72B" in model_name:
                    df.at[idx, "batch_size"] = 4
                    df.at[idx, "num_q_heads"] = 128
                    df.at[idx, "num_kv_heads"] = 8
                    df.at[idx, "head_dim"] = 128
                    df.at[idx, "seq_len"] = 1024

                # Calculate equivalent num_query_items for comparison with PAL
                df.at[idx, "num_query_items"] = df.at[idx, "batch_size"] * df.at[idx, "num_q_heads"]

                # Calculate effective_items for comparison
                df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

        # Handle Python PAL benchmarks
        elif source == "python_pal":
            if base_name == "test_pal_latency_vs_seq_len":
                # Format: test_pal_latency_vs_seq_len[64]
                if params_str:
                    df.at[idx, "seq_len"] = int(params_str)

                    # Set defaults for other parameters
                    df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                    df.at[idx, "num_query_items"] = DEFAULT_NUM_QUERY_ITEMS
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                    df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )

            elif base_name == "test_pal_latency_vs_head_dim":
                # Format: test_pal_latency_vs_head_dim[128]
                if params_str:
                    df.at[idx, "head_dim"] = int(params_str)

                    # Set defaults for other parameters
                    df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                    df.at[idx, "num_query_items"] = DEFAULT_NUM_QUERY_ITEMS
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                    df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )

            elif base_name == "test_pal_latency_vs_query_items":
                # Format: test_pal_latency_vs_query_items[64]
                if params_str:
                    df.at[idx, "num_query_items"] = int(params_str)

                    # Set defaults for other parameters
                    df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                    df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                    df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )

            elif base_name == "test_pal_latency_model_configs":
                # Set model config name
                if row["model_config_name_raw"] is not None:
                    df.at[idx, "model_config_name"] = row["model_config_name_raw"]

                # Parse model params if available
                if row["model_params_raw"] is not None:
                    try:
                        # Parse from JSON string, or use directly if it's already a dict
                        if isinstance(row["model_params_raw"], str):
                            model_params = json.loads(row["model_params_raw"])
                        else:
                            model_params = row["model_params_raw"]

                        if isinstance(model_params, dict):
                            # Extract common parameters from model_params
                            df.at[idx, "num_query_items"] = model_params.get(
                                "num_query_items", df.at[idx, "num_query_items"]
                            )
                            df.at[idx, "num_q_heads"] = model_params.get(
                                "num_q_heads", model_params.get("nqh", df.at[idx, "num_q_heads"])
                            )
                            df.at[idx, "num_kv_heads"] = model_params.get(
                                "num_kv_heads", model_params.get("nkvh", df.at[idx, "num_kv_heads"])
                            )
                            df.at[idx, "head_dim"] = model_params.get(
                                "head_dim", model_params.get("hd", df.at[idx, "head_dim"])
                            )
                            df.at[idx, "seq_len"] = model_params.get(
                                "seq_len", model_params.get("sl", df.at[idx, "seq_len"])
                            )

                            # Additional parameters that might be in model_params
                            df.at[idx, "tokens_per_page"] = model_params.get("tokens_per_page", DEFAULT_TOKENS_PER_PAGE)
                            df.at[idx, "num_sequences_in_batch"] = model_params.get(
                                "num_sequences_in_batch", DEFAULT_NUM_SEQUENCES_IN_BATCH
                            )

                            # Calculate effective_items for comparison
                            df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

                            # Calculate throughput if mean_latency_ms is available
                            if (
                                pd.notna(row["mean_latency_ms"])
                                and row["mean_latency_ms"] > 0
                                and pd.notna(df.at[idx, "num_query_items"])
                            ):
                                df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                                    row["mean_latency_ms"] / 1000.0
                                )

                            # Print for debugging
                            print(
                                f"Extracted PAL model params for {row['model_config_name_raw']}: "
                                + f"num_query_items={df.at[idx, 'num_query_items']}, "
                                + f"num_q_heads={df.at[idx, 'num_q_heads']}, "
                                + f"num_kv_heads={df.at[idx, 'num_kv_heads']}, "
                                + f"head_dim={df.at[idx, 'head_dim']}, "
                                + f"seq_len={df.at[idx, 'seq_len']}, "
                                + f"tokens_per_page={df.at[idx, 'tokens_per_page']}, "
                                + f"num_sequences_in_batch={df.at[idx, 'num_sequences_in_batch']}"
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Error parsing model_params_raw for model_configs row {idx}: {e}")

        # Handle Python SDPA benchmarks
        elif source == "python_sdpa":
            if base_name == "test_sdpa_latency_vs_seq_len":
                # Format: test_sdpa_latency_vs_seq_len[64]
                if params_str:
                    df.at[idx, "seq_len"] = int(params_str)

                    # Set defaults for other parameters
                    df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "batch_size"] = DEFAULT_BATCH_SIZE

                    # Calculate total number of query items (for comparative metrics)
                    # In SDPA: batch_size * num_q_heads * seq_len items are processed
                    # But to compare with PAL, we use equivalent of num_query_items = batch_size * num_q_heads
                    df.at[idx, "num_query_items"] = DEFAULT_BATCH_SIZE * DEFAULT_NUM_Q_HEADS

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )

            elif base_name == "test_sdpa_latency_vs_head_dim":
                # Format: test_sdpa_latency_vs_head_dim[128]
                if params_str:
                    df.at[idx, "head_dim"] = int(params_str)

                    # Set defaults for other parameters
                    df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "batch_size"] = DEFAULT_BATCH_SIZE

                    # Calculate total number of query items (for comparative metrics)
                    df.at[idx, "num_query_items"] = DEFAULT_BATCH_SIZE * DEFAULT_NUM_Q_HEADS

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )

            elif base_name == "test_sdpa_latency_vs_batch_size":
                # Format: test_sdpa_latency_vs_batch_size[64]
                if params_str:
                    batch_size = int(params_str)
                    df.at[idx, "batch_size"] = batch_size

                    # Set defaults for other parameters
                    df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                    df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS

                    # Calculate total number of query items (for comparative metrics)
                    df.at[idx, "num_query_items"] = batch_size * DEFAULT_NUM_Q_HEADS

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )

            elif base_name == "test_sdpa_latency_model_configs":
                # Set model config name
                if row["model_config_name_raw"] is not None:
                    df.at[idx, "model_config_name"] = row["model_config_name_raw"]

                # Parse model params if available
                if row["model_params_raw"] is not None:
                    try:
                        # Parse from JSON string, or use directly if it's already a dict
                        if isinstance(row["model_params_raw"], str):
                            model_params = json.loads(row["model_params_raw"])
                        else:
                            model_params = row["model_params_raw"]

                        if isinstance(model_params, dict):
                            # Extract common parameters from model_params
                            df.at[idx, "batch_size"] = model_params.get("batch_size", DEFAULT_BATCH_SIZE)
                            df.at[idx, "num_q_heads"] = model_params.get(
                                "num_q_heads", model_params.get("nqh", df.at[idx, "num_q_heads"])
                            )
                            df.at[idx, "num_kv_heads"] = model_params.get(
                                "num_kv_heads", model_params.get("nkvh", df.at[idx, "num_kv_heads"])
                            )
                            df.at[idx, "head_dim"] = model_params.get(
                                "head_dim", model_params.get("hd", df.at[idx, "head_dim"])
                            )
                            df.at[idx, "seq_len"] = model_params.get(
                                "seq_len", model_params.get("sl", df.at[idx, "seq_len"])
                            )

                            # Calculate total number of query items (for comparative metrics)
                            batch_size = df.at[idx, "batch_size"]
                            num_q_heads = df.at[idx, "num_q_heads"]
                            df.at[idx, "num_query_items"] = (
                                batch_size * num_q_heads if pd.notna(batch_size) and pd.notna(num_q_heads) else None
                            )

                            # Calculate effective_items for comparison
                            df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]

                            # Calculate throughput if mean_latency_ms is available
                            if (
                                pd.notna(row["mean_latency_ms"])
                                and row["mean_latency_ms"] > 0
                                and pd.notna(df.at[idx, "num_query_items"])
                            ):
                                df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                                    row["mean_latency_ms"] / 1000.0
                                )

                            # Print for debugging
                            print(
                                f"Extracted SDPA model params for {row['model_config_name_raw']}: "
                                + f"batch_size={df.at[idx, 'batch_size']}, "
                                + f"num_q_heads={df.at[idx, 'num_q_heads']}, "
                                + f"num_kv_heads={df.at[idx, 'num_kv_heads']}, "
                                + f"head_dim={df.at[idx, 'head_dim']}, "
                                + f"seq_len={df.at[idx, 'seq_len']}"
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"Error parsing model_params_raw for SDPA model_configs row {idx}: {e}")

    # Convert parameter columns to appropriate numeric types
    for col in [
        "seq_len",
        "head_dim",
        "num_query_items",
        "num_q_heads",
        "num_kv_heads",
        "tokens_per_page",
        "num_sequences_in_batch",
        "batch_size",
        "effective_items",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results from Python and C++ benchmarks.")
    parser.add_argument(
        "--results-dir", type=Path, default=Path(".benchmarks"), help="Directory containing benchmark JSON output files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".benchmarks"), help="Directory to save analysis results"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files in the results directory
    json_files = list(args.results_dir.glob("*_results.json"))

    if not json_files:
        print(f"No benchmark result files found in {args.results_dir}")
        return

    print(f"Found {len(json_files)} benchmark result files")
    for json_file in json_files:
        print(f"  - {json_file}")

    # Load results from all files
    df = load_results(json_files)

    if df.empty:
        print("No benchmark results parsed successfully")
        return

    # Extract and normalize parameters
    df = extract_and_normalize_parameters(df)

    # Print DataFrame information for debugging
    print("\nBenchmark DataFrame summary:")
    print(df[["benchmark_name_base", "source", "mean_latency_ms"]].groupby(["benchmark_name_base", "source"]).count())

    # Dictionary to track generated plot filenames
    plot_filenames = {}

    # Generate consolidated plots
    print("\nGenerating consolidated comparison plots...")

    # Latency vs Sequence Length - the most important plot
    seq_len_filename = plot_latency_vs_seq_len(df, args.output_dir)
    if seq_len_filename:
        plot_filenames["latency_vs_seq_len"] = seq_len_filename

    # Latency vs Head Dimension
    head_dim_filename = plot_latency_vs_head_dim(df, args.output_dir)
    if head_dim_filename:
        plot_filenames["latency_vs_head_dim"] = head_dim_filename

    # Latency vs Number of Items
    num_items_filename = plot_latency_vs_num_query_items(df, args.output_dir)
    if num_items_filename:
        plot_filenames["latency_vs_num_items"] = num_items_filename

    # Model configurations
    model_config_filenames = plot_model_configs_latency(df, args.output_dir)
    if model_config_filenames:
        plot_filenames.update(model_config_filenames)

    # Generate JSON report with all metrics
    print("\nGenerating JSON report with metrics and plot filenames...")
    generate_json_report(df, args.output_dir, plot_filenames)

    print(f"\nAnalysis complete. All results saved to {args.output_dir}")
    print(f"✅ Generated plots: {', '.join(plot_filenames.values())}")
    print("✅ Generated JSON report: results.json")


if __name__ == "__main__":
    main()
