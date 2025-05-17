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

    # Extract source information from filename
    filename = json_file.name

    # Default source if not determinable from filename
    source = "cpp_unknown"

    # Parse source from new filename pattern: {target}_{benchmark_name}_{timestamp}.json
    if filename.startswith("cpp_pal_") or "_cpp_pal_" in filename:
        source = "cpp_pal"
    elif filename.startswith("cpp_sdpa_") or "_cpp_sdpa_" in filename:
        source = "cpp_sdpa"
    elif filename.startswith("cpp_all_") or "_cpp_all_" in filename:
        # We'll determine actual source based on benchmark name
        source = "cpp_all"

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

            # If source is cpp_all, determine the actual source based on the benchmark name
            if source == "cpp_all":
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

    # Extract source information from filename
    filename = json_file.name
    print(f"Processing pytest benchmark file: {filename}")

    # Default source if not determinable from filename
    source_from_file = "python_unknown"

    # Parse source from new filename pattern: {target}_{benchmark_name}_{timestamp}.json
    if filename.startswith("py_pal_") or "_py_pal_" in filename or filename.startswith("pal_"):
        source_from_file = "python_pal"
        print(f"  Detected source from filename: {source_from_file}")
    elif filename.startswith("py_sdpa_") or "_py_sdpa_" in filename or filename.startswith("sdpa_"):
        source_from_file = "python_sdpa"
        print(f"  Detected source from filename: {source_from_file}")
    elif filename.startswith("py_all_") or "_py_all_" in filename:
        # We'll determine source from test name
        source_from_file = "python_all"
        print(f"  Will determine source from test name for: {source_from_file}")

    print(f"  Found {len(benchmarks)} benchmark entries in file")

    for bench in benchmarks:
        name = bench.get("name", "")
        print(f"  Processing benchmark: {name}")

        # Extract group as base name if available, otherwise parse from name
        group = bench.get("group", "")
        base_name = group if group else name.split("[")[0] if "[" in name else name
        print(f"    Base name: {base_name}")

        # Determine source based on benchmark name or filename
        if source_from_file == "python_all":
            # Determine source from test name for combined benchmarks
            if base_name.startswith("test_pal_"):
                source = "python_pal"
            elif base_name.startswith("test_sdpa_"):
                source = "python_sdpa"
            else:
                source = "python"  # Default for any other Python benchmarks
            print(f"    Detected source from base_name: {source}")
        else:
            # Use the source determined from filename
            source = source_from_file
            print(f"    Using source from filename: {source}")

        # Extract params from name (inside brackets)
        params_str = ""
        if "[" in name and "]" in name:
            params_str = name.split("[")[1].split("]")[0]
            print(f"    Extracted params: {params_str}")

        # Extract stats
        stats = bench.get("stats", {})
        mean_time_sec = stats.get("mean", 0)
        stddev_sec = stats.get("stddev", 0)

        # Convert to ms
        mean_latency_ms = mean_time_sec * 1000.0
        stddev_latency_ms = stddev_sec * 1000.0
        print(f"    Mean latency: {mean_latency_ms:.4f} ms")

        # Extract model config name if present
        model_config_name_raw = None
        if "model_configs" in base_name and "[" in name and "-model_params" in name:
            config_part = name.split("[")[1].split("-model_params")[0]
            model_config_name_raw = config_part
            print(f"    Model config name: {model_config_name_raw}")

        # Extract model params if present
        model_params_raw = None
        if "params" in bench and "model_params" in bench["params"]:
            model_params_raw = json.dumps(bench["params"]["model_params"])
            print(f"    Has model params: {model_params_raw is not None}")

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

    print(f"  Processed {len(rows)} benchmark rows from file")
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
    print("\nGenerating latency vs. sequence length plot")

    # Print unique benchmark names for debugging
    print("Available benchmark_name_base values:")
    print(df["benchmark_name_base"].unique())

    # Print source counts for debugging
    source_counts = df["source"].value_counts()
    print(f"Source counts:\n{source_counts}")

    # First define our benchmark base name patterns
    pal_py_base = "test_pal_latency_vs_seq_len"
    sdpa_py_base = "test_sdpa_latency_vs_seq_len"
    pal_cpp_base = "BM_PAL_LatencyVsSeqLen"
    sdpa_cpp_base = "BM_SDPA_LatencyVsSeqLen"

    # Filter by benchmark base name patterns
    seq_len_df = df[df["benchmark_name_base"].isin([pal_py_base, sdpa_py_base, pal_cpp_base, sdpa_cpp_base])]

    print(f"Found {len(seq_len_df)} total sequence length benchmark rows")
    print(f"Sources in filtered data: {seq_len_df['source'].value_counts().to_dict()}")

    # Now split by source
    pal_cpp_df = seq_len_df[seq_len_df["source"] == "cpp_pal"]
    pal_py_df = seq_len_df[seq_len_df["source"] == "python_pal"]
    sdpa_cpp_df = seq_len_df[seq_len_df["source"] == "cpp_sdpa"]
    sdpa_py_df = seq_len_df[seq_len_df["source"] == "python_sdpa"]

    # Print summary of each dataframe
    print(f"PAL Python data points: {len(pal_py_df)}")
    print(f"SDPA Python data points: {len(sdpa_py_df)}")
    print(f"PAL C++ data points: {len(pal_cpp_df)}")
    print(f"SDPA C++ data points: {len(sdpa_cpp_df)}")

    # Check if we have any data to plot
    has_data = not pal_cpp_df.empty or not pal_py_df.empty or not sdpa_cpp_df.empty or not sdpa_py_df.empty

    if not has_data:
        print("No data for latency vs. sequence length plot")
        return ""

    # Debug: print sequence length values for each implementation
    if not pal_py_df.empty:
        print(f"PAL Python seq_len values: {sorted(pal_py_df['seq_len'].unique())}")
    if not sdpa_py_df.empty:
        print(f"SDPA Python seq_len values: {sorted(sdpa_py_df['seq_len'].unique())}")

    # Create the plot with high-quality settings
    plt.figure(figsize=(12, 8), dpi=100)

    # Define consistent styling for implementations
    # Colors
    PAL_CPP_COLOR = "#024645"  # Dark green for PAL C++
    PAL_PY_COLOR = "#026645"  # Slightly different green for PAL Python
    SDPA_CPP_COLOR = "#000000"  # Black for SDPA C++
    SDPA_PY_COLOR = "#444444"  # Dark gray for SDPA Python for better differentiation

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
        print("Plotted PAL C++ data")

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
        print("Plotted PAL Python data")

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
        print("Plotted SDPA C++ data")

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
        print("Plotted SDPA Python data")

    # Set plot attributes with refined styling
    plt.title("Paged Attention vs. MLX SDPA: Latency vs. Sequence Length", fontsize=16, fontweight="bold")
    plt.xlabel("Sequence Length (tokens)", fontsize=14)
    plt.ylabel("Mean Latency (ms)", fontsize=14)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")
    plt.legend(loc="best", fontsize=12)  # Add legend

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
    print("\nGenerating latency vs. head dimension plot")

    # First define our benchmark base name patterns
    pal_py_base = "test_pal_latency_vs_head_dim"
    sdpa_py_base = "test_sdpa_latency_vs_head_dim"
    pal_cpp_base = "BM_PAL_LatencyVsHeadDim"
    sdpa_cpp_base = "BM_SDPA_LatencyVsHeadDim"

    # Filter by benchmark base name patterns
    head_dim_df = df[df["benchmark_name_base"].isin([pal_py_base, sdpa_py_base, pal_cpp_base, sdpa_cpp_base])]

    print(f"Found {len(head_dim_df)} total head dimension benchmark rows")
    print(f"Sources in filtered data: {head_dim_df['source'].value_counts().to_dict()}")

    # Now split by source
    pal_cpp_df = head_dim_df[head_dim_df["source"] == "cpp_pal"]
    pal_py_df = head_dim_df[head_dim_df["source"] == "python_pal"]
    sdpa_cpp_df = head_dim_df[head_dim_df["source"] == "cpp_sdpa"]
    sdpa_py_df = head_dim_df[head_dim_df["source"] == "python_sdpa"]

    # Print summary of each dataframe
    print(f"PAL Python data points: {len(pal_py_df)}")
    print(f"SDPA Python data points: {len(sdpa_py_df)}")
    print(f"PAL C++ data points: {len(pal_cpp_df)}")
    print(f"SDPA C++ data points: {len(sdpa_cpp_df)}")

    # Check if we have any data to plot
    has_data = not pal_cpp_df.empty or not pal_py_df.empty or not sdpa_cpp_df.empty or not sdpa_py_df.empty

    if not has_data:
        print("No data for latency vs. head dimension plot")
        return ""

    # Debug: print head dimension values for each implementation
    if not pal_py_df.empty:
        print(f"PAL Python head_dim values: {sorted(pal_py_df['head_dim'].unique())}")
    if not sdpa_py_df.empty:
        print(f"SDPA Python head_dim values: {sorted(sdpa_py_df['head_dim'].unique())}")

    # Create the plot with high-quality settings
    plt.figure(figsize=(12, 8), dpi=100)

    # Define consistent styling for implementations
    # Colors
    PAL_CPP_COLOR = "#024645"  # Dark green for PAL C++
    PAL_PY_COLOR = "#026645"  # Slightly different green for PAL Python
    SDPA_CPP_COLOR = "#000000"  # Black for SDPA C++
    SDPA_PY_COLOR = "#444444"  # Dark gray for SDPA Python for better differentiation

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
        print("Plotted PAL C++ data")

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
        print("Plotted PAL Python data")

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
        print("Plotted SDPA C++ data")

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
        print("Plotted SDPA Python data")

    # Set plot attributes with refined styling
    plt.title("Paged Attention vs. MLX SDPA: Latency vs. Head Dimension", fontsize=16, fontweight="bold")
    plt.xlabel("Head Dimension", fontsize=14)
    plt.ylabel("Mean Latency (ms)", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")
    plt.legend(loc="best", fontsize=12)  # Add legend

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
    print("\nGenerating latency vs. query items / batch size plot")

    # First define our benchmark base name patterns
    pal_py_base = "test_pal_latency_vs_query_items"
    sdpa_py_base = "test_sdpa_latency_vs_batch_size"
    pal_cpp_base = "BM_PAL_LatencyVsNumItems"
    sdpa_cpp_base = "BM_SDPA_LatencyVsNumItems"

    # Filter by benchmark base name patterns
    query_items_df = df[df["benchmark_name_base"].isin([pal_py_base, sdpa_py_base, pal_cpp_base, sdpa_cpp_base])]

    print(f"Found {len(query_items_df)} total query items/batch size benchmark rows")
    print(f"Sources in filtered data: {query_items_df['source'].value_counts().to_dict()}")

    # Now split by source
    pal_cpp_df = query_items_df[query_items_df["source"] == "cpp_pal"]
    pal_py_df = query_items_df[query_items_df["source"] == "python_pal"]
    sdpa_cpp_df = query_items_df[query_items_df["source"] == "cpp_sdpa"]
    sdpa_py_df = query_items_df[query_items_df["source"] == "python_sdpa"]

    # Print summary of each dataframe
    print(f"PAL Python data points: {len(pal_py_df)}")
    print(f"SDPA Python data points: {len(sdpa_py_df)}")
    print(f"PAL C++ data points: {len(pal_cpp_df)}")
    print(f"SDPA C++ data points: {len(sdpa_cpp_df)}")

    # Check if we have any data to plot
    has_data = not pal_cpp_df.empty or not pal_py_df.empty or not sdpa_cpp_df.empty or not sdpa_py_df.empty

    if not has_data:
        print("No data for latency vs. query items / batch size plot")
        return ""

    # Debug: print values for each implementation
    if not pal_py_df.empty:
        print(f"PAL Python num_query_items values: {sorted(pal_py_df['num_query_items'].unique())}")
        print(f"PAL Python effective_items values: {sorted(pal_py_df['effective_items'].unique())}")
    if not sdpa_py_df.empty:
        print(f"SDPA Python batch_size values: {sorted(sdpa_py_df['batch_size'].unique())}")
        print(f"SDPA Python effective_items values: {sorted(sdpa_py_df['effective_items'].unique())}")

    # Create the plot with high-quality settings
    plt.figure(figsize=(12, 8), dpi=100)

    # Define consistent styling for implementations
    # Colors
    PAL_CPP_COLOR = "#024645"  # Dark green for PAL C++
    PAL_PY_COLOR = "#026645"  # Slightly different green for PAL Python
    SDPA_CPP_COLOR = "#000000"  # Black for SDPA C++
    SDPA_PY_COLOR = "#444444"  # Dark gray for SDPA Python for better differentiation

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
        print("Plotted PAL C++ data")

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
        print("Plotted PAL Python data")

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
        print("Plotted SDPA C++ data")

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
        print("Plotted SDPA Python data")

    # Set plot attributes with refined styling
    plt.title("Paged Attention vs. MLX SDPA: Latency vs. Processing Items", fontsize=16, fontweight="bold")
    plt.xlabel("Number of Effective Items (PAL: Query Items, SDPA: Batch Size x Heads)", fontsize=14)
    plt.ylabel("Mean Latency (ms)", fontsize=14)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2, color="lightgray")
    plt.legend(loc="best", fontsize=12)  # Add legend

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
    print("\nGenerating model configurations latency plot")

    # Dictionary to track all plot filenames created
    plot_filenames = {}

    # Define our benchmark base name patterns
    pal_py_base = "test_pal_latency_model_configs"
    sdpa_py_base = "test_sdpa_latency_model_configs"

    # Get model config benchmark dataframes for each implementation
    pal_py_df = df[(df["source"] == "python_pal") & (df["benchmark_name_base"] == pal_py_base)]
    pal_cpp_df = df[(df["source"] == "cpp_pal") & (df["benchmark_name_base"].str.contains("ModelConfig"))]
    sdpa_py_df = df[(df["source"] == "python_sdpa") & (df["benchmark_name_base"] == sdpa_py_base)]
    sdpa_cpp_df = df[(df["source"] == "cpp_sdpa") & (df["benchmark_name_base"].str.contains("ModelConfig"))]

    # Print summary of available data
    print(f"PAL Python model configs: {len(pal_py_df)}")
    print(f"SDPA Python model configs: {len(sdpa_py_df)}")
    print(f"PAL C++ model configs: {len(pal_cpp_df)}")
    print(f"SDPA C++ model configs: {len(sdpa_cpp_df)}")

    # Create a consolidated comparison plot across all available implementations
    model_configs_to_plot = set()

    if not pal_py_df.empty:
        config_names = pal_py_df["model_config_name"].dropna().unique()
        print(f"PAL Python model config names: {config_names}")
        model_configs_to_plot.update(config_names)

    if not pal_cpp_df.empty:
        config_names = pal_cpp_df["model_config_name"].dropna().unique()
        print(f"PAL C++ model config names: {config_names}")
        model_configs_to_plot.update(config_names)

    if not sdpa_py_df.empty:
        config_names = sdpa_py_df["model_config_name"].dropna().unique()
        print(f"SDPA Python model config names: {config_names}")
        model_configs_to_plot.update(config_names)

    if not sdpa_cpp_df.empty:
        config_names = sdpa_cpp_df["model_config_name"].dropna().unique()
        print(f"SDPA C++ model config names: {config_names}")
        model_configs_to_plot.update(config_names)

    # Skip if no config data available
    if not model_configs_to_plot:
        print("No model configuration benchmark data available")
        return plot_filenames

    print(f"Found {len(model_configs_to_plot)} unique model configurations to plot")

    # Prepare data for plot - we'll create a wide-format DataFrame
    plot_data = []

    for config in sorted(model_configs_to_plot):
        row_data = {"model_config_name": config}
        print(f"Processing config: {config}")

        # Add PAL Python latency if available
        if not pal_py_df.empty:
            config_data = pal_py_df[pal_py_df["model_config_name"] == config]
            if not config_data.empty:
                row_data["pal_py_latency_ms"] = config_data["mean_latency_ms"].iloc[0]
                print(f"  PAL Python latency: {row_data['pal_py_latency_ms']:.2f} ms")

        # Add PAL C++ latency if available
        if not pal_cpp_df.empty:
            config_data = pal_cpp_df[pal_cpp_df["model_config_name"] == config]
            if not config_data.empty:
                row_data["pal_cpp_latency_ms"] = config_data["mean_latency_ms"].iloc[0]
                print(f"  PAL C++ latency: {row_data['pal_cpp_latency_ms']:.2f} ms")

        # Add SDPA Python latency if available
        if not sdpa_py_df.empty:
            config_data = sdpa_py_df[sdpa_py_df["model_config_name"] == config]
            if not config_data.empty:
                row_data["sdpa_py_latency_ms"] = config_data["mean_latency_ms"].iloc[0]
                print(f"  SDPA Python latency: {row_data['sdpa_py_latency_ms']:.2f} ms")

        # Add SDPA C++ latency if available
        if not sdpa_cpp_df.empty:
            config_data = sdpa_cpp_df[sdpa_cpp_df["model_config_name"] == config]
            if not config_data.empty:
                row_data["sdpa_cpp_latency_ms"] = config_data["mean_latency_ms"].iloc[0]
                print(f"  SDPA C++ latency: {row_data['sdpa_cpp_latency_ms']:.2f} ms")

        # Add speedup ratios where possible
        # PAL Python / SDPA Python
        if "pal_py_latency_ms" in row_data and "sdpa_py_latency_ms" in row_data and row_data["sdpa_py_latency_ms"] > 0:
            row_data["pal_py_sdpa_py_ratio"] = row_data["pal_py_latency_ms"] / row_data["sdpa_py_latency_ms"]
            print(f"  PAL Python / SDPA Python ratio: {row_data['pal_py_sdpa_py_ratio']:.2f}x")

        # PAL C++ / SDPA C++
        if (
            "pal_cpp_latency_ms" in row_data
            and "sdpa_cpp_latency_ms" in row_data
            and row_data["sdpa_cpp_latency_ms"] > 0
        ):
            row_data["pal_cpp_sdpa_cpp_ratio"] = row_data["pal_cpp_latency_ms"] / row_data["sdpa_cpp_latency_ms"]
            print(f"  PAL C++ / SDPA C++ ratio: {row_data['pal_cpp_sdpa_cpp_ratio']:.2f}x")

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

    # Debug print the final data we'll be plotting
    print("\nFinal data for plotting:")
    print(plot_df)

    # Figure out which columns are available for plotting
    available_columns = [
        col
        for col in ["pal_py_latency_ms", "pal_cpp_latency_ms", "sdpa_py_latency_ms", "sdpa_cpp_latency_ms"]
        if col in plot_df.columns
    ]

    print(f"Available columns for plotting: {available_columns}")

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
            # Check if we have any non-NaN values for this column
            if not plot_df[col].notna().any():
                print(f"Warning: No valid data for {col}, skipping...")
                continue

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

            print(f"Plotted bars for {col}")

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

        # Add legend
        plt.legend(fontsize=10)

        # Add speedup ratios for Python implementations
        if "pal_py_sdpa_py_ratio" in plot_df.columns and plot_df["pal_py_sdpa_py_ratio"].notna().any():
            print("Adding Python implementation speedup ratios")
            for i, (ratio, is_valid) in enumerate(
                zip(plot_df["pal_py_sdpa_py_ratio"], plot_df["pal_py_sdpa_py_ratio"].notna(), strict=True)
            ):
                if is_valid:
                    # Find the maximum bar height at this position
                    bar_heights = []
                    for bars_collection in all_bars.values():
                        if i < len(bars_collection):
                            bar = bars_collection[i]
                            bar_heights.append(bar.get_height())

                    max_height = max(bar_heights) if bar_heights else 0
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
        if "pal_cpp_sdpa_cpp_ratio" in plot_df.columns and plot_df["pal_cpp_sdpa_cpp_ratio"].notna().any():
            print("Adding C++ implementation speedup ratios")
            for i, (ratio, is_valid) in enumerate(
                zip(plot_df["pal_cpp_sdpa_cpp_ratio"], plot_df["pal_cpp_sdpa_cpp_ratio"].notna(), strict=True)
            ):
                if is_valid:
                    # Find the maximum bar height at this position
                    bar_heights = []
                    for bars_collection in all_bars.values():
                        if i < len(bars_collection):
                            bar = bars_collection[i]
                            bar_heights.append(bar.get_height())

                    max_height = max(bar_heights) if bar_heights else 0
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

    print("Generating JSON report with summary metrics and plot filenames")

    # Create the report structure
    report = {
        "summary_metrics": {},
        "plot_files": plot_filenames,
        "generation_timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Process latency vs sequence length data
    seq_len_data = []

    # Define benchmark base names
    pal_cpp_seq_base = "BM_PAL_LatencyVsSeqLen"
    pal_py_seq_base = "test_pal_latency_vs_seq_len"
    sdpa_cpp_seq_base = "BM_SDPA_LatencyVsSeqLen"
    sdpa_py_seq_base = "test_sdpa_latency_vs_seq_len"

    # Get sequence length data for all implementations
    seq_len_benchmarks = df[
        df["benchmark_name_base"].isin([pal_cpp_seq_base, pal_py_seq_base, sdpa_cpp_seq_base, sdpa_py_seq_base])
    ]

    print(f"Found {len(seq_len_benchmarks)} sequence length data points for JSON report")

    # Process each implementation separately for clear reporting
    # PAL C++ data
    pal_cpp_seq_df = seq_len_benchmarks[
        (seq_len_benchmarks["source"] == "cpp_pal") & (seq_len_benchmarks["benchmark_name_base"] == pal_cpp_seq_base)
    ]
    if not pal_cpp_seq_df.empty:
        print(f"Adding {len(pal_cpp_seq_df)} PAL C++ sequence length data points to report")
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
    pal_py_seq_df = seq_len_benchmarks[
        (seq_len_benchmarks["source"] == "python_pal") & (seq_len_benchmarks["benchmark_name_base"] == pal_py_seq_base)
    ]
    if not pal_py_seq_df.empty:
        print(f"Adding {len(pal_py_seq_df)} PAL Python sequence length data points to report")
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
    sdpa_cpp_seq_df = seq_len_benchmarks[
        (seq_len_benchmarks["source"] == "cpp_sdpa") & (seq_len_benchmarks["benchmark_name_base"] == sdpa_cpp_seq_base)
    ]
    if not sdpa_cpp_seq_df.empty:
        print(f"Adding {len(sdpa_cpp_seq_df)} SDPA C++ sequence length data points to report")
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
    sdpa_py_seq_df = seq_len_benchmarks[
        (seq_len_benchmarks["source"] == "python_sdpa")
        & (seq_len_benchmarks["benchmark_name_base"] == sdpa_py_seq_base)
    ]
    if not sdpa_py_seq_df.empty:
        print(f"Adding {len(sdpa_py_seq_df)} SDPA Python sequence length data points to report")
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
    print("\nExtracting and normalizing benchmark parameters...")
    print(f"Initial DataFrame contains {len(df)} rows")

    # Count sources in initial data
    source_counts = df["source"].value_counts().to_dict()
    print(f"Source distribution: {source_counts}")

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

    # Track some statistics for reporting
    stats = {
        "python_pal": {"processed": 0, "seq_len": 0, "head_dim": 0, "query_items": 0, "model_configs": 0},
        "python_sdpa": {"processed": 0, "seq_len": 0, "head_dim": 0, "batch_size": 0, "model_configs": 0},
        "cpp_pal": {"processed": 0},
        "cpp_sdpa": {"processed": 0},
    }

    # Process each row
    for idx, row in df.iterrows():
        print(f"\nProcessing row {idx}: {row['benchmark_name_base']} (source: {row['source']})")
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
            print(f"  Processing Python PAL benchmark: {base_name}")
            stats["python_pal"]["processed"] += 1

            if base_name == "test_pal_latency_vs_seq_len":
                # Format: test_pal_latency_vs_seq_len[64]
                print(f"  PAL latency vs seq_len benchmark with params: {params_str}")
                stats["python_pal"]["seq_len"] += 1

                if params_str:
                    seq_len = int(params_str)
                    df.at[idx, "seq_len"] = seq_len
                    print(f"  Extracted seq_len: {seq_len}")

                    # Set defaults for other parameters
                    df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                    df.at[idx, "num_query_items"] = DEFAULT_NUM_QUERY_ITEMS
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                    df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                    print(f"  Set effective_items: {df.at[idx, 'effective_items']}")

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )
                        print(f"  Calculated throughput: {df.at[idx, 'throughput_items_per_sec']:.2f} items/sec")

            elif base_name == "test_pal_latency_vs_head_dim":
                # Format: test_pal_latency_vs_head_dim[128]
                print(f"  PAL latency vs head_dim benchmark with params: {params_str}")
                stats["python_pal"]["head_dim"] += 1

                if params_str:
                    head_dim = int(params_str)
                    df.at[idx, "head_dim"] = head_dim
                    print(f"  Extracted head_dim: {head_dim}")

                    # Set defaults for other parameters
                    df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                    df.at[idx, "num_query_items"] = DEFAULT_NUM_QUERY_ITEMS
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                    df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                    print(f"  Set effective_items: {df.at[idx, 'effective_items']}")

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )
                        print(f"  Calculated throughput: {df.at[idx, 'throughput_items_per_sec']:.2f} items/sec")

            elif base_name == "test_pal_latency_vs_query_items":
                # Format: test_pal_latency_vs_query_items[64]
                print(f"  PAL latency vs query_items benchmark with params: {params_str}")
                stats["python_pal"]["query_items"] += 1

                if params_str:
                    num_query_items = int(params_str)
                    df.at[idx, "num_query_items"] = num_query_items
                    print(f"  Extracted num_query_items: {num_query_items}")

                    # Set defaults for other parameters
                    df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                    df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                    df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                    print(f"  Set effective_items: {df.at[idx, 'effective_items']}")

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )
                        print(f"  Calculated throughput: {df.at[idx, 'throughput_items_per_sec']:.2f} items/sec")

            elif base_name == "test_pal_latency_model_configs":
                print("  PAL model config benchmark")
                stats["python_pal"]["model_configs"] += 1

                # Set model config name
                if row["model_config_name_raw"] is not None:
                    df.at[idx, "model_config_name"] = row["model_config_name_raw"]
                    print(f"  Model config name: {row['model_config_name_raw']}")

                # Parse model params if available
                if row["model_params_raw"] is not None:
                    try:
                        # Parse from JSON string, or use directly if it's already a dict
                        if isinstance(row["model_params_raw"], str):
                            model_params = json.loads(row["model_params_raw"])
                            print("  Parsed model_params from JSON string")
                        else:
                            model_params = row["model_params_raw"]
                            print("  Using existing model_params dict")

                        if isinstance(model_params, dict):
                            print(f"  Model params keys: {list(model_params.keys())}")

                            # Extract common parameters from model_params
                            if "num_query_items" in model_params:
                                df.at[idx, "num_query_items"] = model_params["num_query_items"]
                                print(f"  From params -> num_query_items: {df.at[idx, 'num_query_items']}")

                            # num_q_heads can be in different formats
                            if "num_q_heads" in model_params:
                                df.at[idx, "num_q_heads"] = model_params["num_q_heads"]
                                print(f"  From params -> num_q_heads: {df.at[idx, 'num_q_heads']}")
                            elif "nqh" in model_params:
                                df.at[idx, "num_q_heads"] = model_params["nqh"]
                                print(f"  From params -> nqh: {df.at[idx, 'num_q_heads']}")

                            # num_kv_heads can be in different formats
                            if "num_kv_heads" in model_params:
                                df.at[idx, "num_kv_heads"] = model_params["num_kv_heads"]
                                print(f"  From params -> num_kv_heads: {df.at[idx, 'num_kv_heads']}")
                            elif "nkvh" in model_params:
                                df.at[idx, "num_kv_heads"] = model_params["nkvh"]
                                print(f"  From params -> nkvh: {df.at[idx, 'num_kv_heads']}")

                            # head_dim can be in different formats
                            if "head_dim" in model_params:
                                df.at[idx, "head_dim"] = model_params["head_dim"]
                                print(f"  From params -> head_dim: {df.at[idx, 'head_dim']}")
                            elif "hd" in model_params:
                                df.at[idx, "head_dim"] = model_params["hd"]
                                print(f"  From params -> hd: {df.at[idx, 'head_dim']}")

                            # seq_len can be in different formats
                            if "seq_len" in model_params:
                                df.at[idx, "seq_len"] = model_params["seq_len"]
                                print(f"  From params -> seq_len: {df.at[idx, 'seq_len']}")
                            elif "sl" in model_params:
                                df.at[idx, "seq_len"] = model_params["sl"]
                                print(f"  From params -> sl: {df.at[idx, 'seq_len']}")

                            # Additional parameters that might be in model_params
                            if "tokens_per_page" in model_params:
                                df.at[idx, "tokens_per_page"] = model_params["tokens_per_page"]
                                print(f"  From params -> tokens_per_page: {df.at[idx, 'tokens_per_page']}")
                            else:
                                df.at[idx, "tokens_per_page"] = DEFAULT_TOKENS_PER_PAGE
                                print(f"  Using default tokens_per_page: {DEFAULT_TOKENS_PER_PAGE}")

                            if "num_sequences_in_batch" in model_params:
                                df.at[idx, "num_sequences_in_batch"] = model_params["num_sequences_in_batch"]
                                print(
                                    f"  From params -> num_sequences_in_batch: {df.at[idx, 'num_sequences_in_batch']}"
                                )
                            else:
                                df.at[idx, "num_sequences_in_batch"] = DEFAULT_NUM_SEQUENCES_IN_BATCH
                                print(f"  Using default num_sequences_in_batch: {DEFAULT_NUM_SEQUENCES_IN_BATCH}")

                            # Calculate effective_items for comparison
                            df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                            print(f"  Set effective_items: {df.at[idx, 'effective_items']}")

                            # Calculate throughput if mean_latency_ms is available
                            if (
                                pd.notna(row["mean_latency_ms"])
                                and row["mean_latency_ms"] > 0
                                and pd.notna(df.at[idx, "num_query_items"])
                            ):
                                df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                                    row["mean_latency_ms"] / 1000.0
                                )
                                print(
                                    f"  Calculated throughput: {df.at[idx, 'throughput_items_per_sec']:.2f} items/sec"
                                )

                            # Print all extracted params for debugging
                            print(
                                f"  Extracted PAL model params for {row['model_config_name_raw']}: "
                                + f"num_query_items={df.at[idx, 'num_query_items']}, "
                                + f"num_q_heads={df.at[idx, 'num_q_heads']}, "
                                + f"num_kv_heads={df.at[idx, 'num_kv_heads']}, "
                                + f"head_dim={df.at[idx, 'head_dim']}, "
                                + f"seq_len={df.at[idx, 'seq_len']}, "
                                + f"tokens_per_page={df.at[idx, 'tokens_per_page']}, "
                                + f"num_sequences_in_batch={df.at[idx, 'num_sequences_in_batch']}"
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"  ERROR parsing model_params_raw for model_configs row {idx}: {e}")
                        print(f"  Raw value: {row['model_params_raw']}")

        # Handle Python SDPA benchmarks
        elif source == "python_sdpa":
            print(f"  Processing Python SDPA benchmark: {base_name}")
            stats["python_sdpa"]["processed"] += 1

            if base_name == "test_sdpa_latency_vs_seq_len":
                # Format: test_sdpa_latency_vs_seq_len[64]
                print(f"  SDPA latency vs seq_len benchmark with params: {params_str}")
                stats["python_sdpa"]["seq_len"] += 1

                if params_str:
                    seq_len = int(params_str)
                    df.at[idx, "seq_len"] = seq_len
                    print(f"  Extracted seq_len: {seq_len}")

                    # Set defaults for other parameters
                    df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "batch_size"] = DEFAULT_BATCH_SIZE
                    print(f"  Using default batch_size: {DEFAULT_BATCH_SIZE}")

                    # Calculate total number of query items (for comparative metrics)
                    # In SDPA: batch_size * num_q_heads * seq_len items are processed
                    # But to compare with PAL, we use equivalent of num_query_items = batch_size * num_q_heads
                    df.at[idx, "num_query_items"] = DEFAULT_BATCH_SIZE * DEFAULT_NUM_Q_HEADS
                    print(f"  Calculated num_query_items: {df.at[idx, 'num_query_items']}")

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                    print(f"  Set effective_items: {df.at[idx, 'effective_items']}")

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )
                        print(f"  Calculated throughput: {df.at[idx, 'throughput_items_per_sec']:.2f} items/sec")

            elif base_name == "test_sdpa_latency_vs_head_dim":
                # Format: test_sdpa_latency_vs_head_dim[128]
                print(f"  SDPA latency vs head_dim benchmark with params: {params_str}")
                stats["python_sdpa"]["head_dim"] += 1

                if params_str:
                    head_dim = int(params_str)
                    df.at[idx, "head_dim"] = head_dim
                    print(f"  Extracted head_dim: {head_dim}")

                    # Set defaults for other parameters
                    df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS
                    df.at[idx, "batch_size"] = DEFAULT_BATCH_SIZE
                    print(f"  Using default batch_size: {DEFAULT_BATCH_SIZE}")

                    # Calculate total number of query items (for comparative metrics)
                    df.at[idx, "num_query_items"] = DEFAULT_BATCH_SIZE * DEFAULT_NUM_Q_HEADS
                    print(f"  Calculated num_query_items: {df.at[idx, 'num_query_items']}")

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                    print(f"  Set effective_items: {df.at[idx, 'effective_items']}")

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )
                        print(f"  Calculated throughput: {df.at[idx, 'throughput_items_per_sec']:.2f} items/sec")

            elif base_name == "test_sdpa_latency_vs_batch_size":
                # Format: test_sdpa_latency_vs_batch_size[64]
                print(f"  SDPA latency vs batch_size benchmark with params: {params_str}")
                stats["python_sdpa"]["batch_size"] += 1

                if params_str:
                    batch_size = int(params_str)
                    df.at[idx, "batch_size"] = batch_size
                    print(f"  Extracted batch_size: {batch_size}")

                    # Set defaults for other parameters
                    df.at[idx, "head_dim"] = DEFAULT_HEAD_DIM
                    df.at[idx, "seq_len"] = DEFAULT_SEQ_LEN
                    df.at[idx, "num_q_heads"] = DEFAULT_NUM_Q_HEADS
                    df.at[idx, "num_kv_heads"] = DEFAULT_NUM_KV_HEADS

                    # Calculate total number of query items (for comparative metrics)
                    df.at[idx, "num_query_items"] = batch_size * DEFAULT_NUM_Q_HEADS
                    print(f"  Calculated num_query_items: {df.at[idx, 'num_query_items']}")

                    # Calculate effective_items for comparison
                    df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                    print(f"  Set effective_items: {df.at[idx, 'effective_items']}")

                    # Calculate throughput if mean_latency_ms is available
                    if pd.notna(row["mean_latency_ms"]) and row["mean_latency_ms"] > 0:
                        df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                            row["mean_latency_ms"] / 1000.0
                        )
                        print(f"  Calculated throughput: {df.at[idx, 'throughput_items_per_sec']:.2f} items/sec")

            elif base_name == "test_sdpa_latency_model_configs":
                print("  SDPA model config benchmark")
                stats["python_sdpa"]["model_configs"] += 1

                # Set model config name
                if row["model_config_name_raw"] is not None:
                    df.at[idx, "model_config_name"] = row["model_config_name_raw"]
                    print(f"  Model config name: {row['model_config_name_raw']}")

                # Parse model params if available
                if row["model_params_raw"] is not None:
                    try:
                        # Parse from JSON string, or use directly if it's already a dict
                        if isinstance(row["model_params_raw"], str):
                            model_params = json.loads(row["model_params_raw"])
                            print("  Parsed model_params from JSON string")
                        else:
                            model_params = row["model_params_raw"]
                            print("  Using existing model_params dict")

                        if isinstance(model_params, dict):
                            print(f"  Model params keys: {list(model_params.keys())}")

                            # Extract batch_size
                            if "batch_size" in model_params:
                                df.at[idx, "batch_size"] = model_params["batch_size"]
                                print(f"  From params -> batch_size: {df.at[idx, 'batch_size']}")
                            else:
                                df.at[idx, "batch_size"] = DEFAULT_BATCH_SIZE
                                print(f"  Using default batch_size: {DEFAULT_BATCH_SIZE}")

                            # num_q_heads can be in different formats
                            if "num_q_heads" in model_params:
                                df.at[idx, "num_q_heads"] = model_params["num_q_heads"]
                                print(f"  From params -> num_q_heads: {df.at[idx, 'num_q_heads']}")
                            elif "nqh" in model_params:
                                df.at[idx, "num_q_heads"] = model_params["nqh"]
                                print(f"  From params -> nqh: {df.at[idx, 'num_q_heads']}")

                            # num_kv_heads can be in different formats
                            if "num_kv_heads" in model_params:
                                df.at[idx, "num_kv_heads"] = model_params["num_kv_heads"]
                                print(f"  From params -> num_kv_heads: {df.at[idx, 'num_kv_heads']}")
                            elif "nkvh" in model_params:
                                df.at[idx, "num_kv_heads"] = model_params["nkvh"]
                                print(f"  From params -> nkvh: {df.at[idx, 'num_kv_heads']}")

                            # head_dim can be in different formats
                            if "head_dim" in model_params:
                                df.at[idx, "head_dim"] = model_params["head_dim"]
                                print(f"  From params -> head_dim: {df.at[idx, 'head_dim']}")
                            elif "hd" in model_params:
                                df.at[idx, "head_dim"] = model_params["hd"]
                                print(f"  From params -> hd: {df.at[idx, 'head_dim']}")

                            # seq_len can be in different formats
                            if "seq_len" in model_params:
                                df.at[idx, "seq_len"] = model_params["seq_len"]
                                print(f"  From params -> seq_len: {df.at[idx, 'seq_len']}")
                            elif "sl" in model_params:
                                df.at[idx, "seq_len"] = model_params["sl"]
                                print(f"  From params -> sl: {df.at[idx, 'seq_len']}")

                            # Calculate total number of query items (for comparative metrics)
                            batch_size = df.at[idx, "batch_size"]
                            num_q_heads = df.at[idx, "num_q_heads"]
                            if pd.notna(batch_size) and pd.notna(num_q_heads):
                                df.at[idx, "num_query_items"] = batch_size * num_q_heads
                                print(f"  Calculated num_query_items: {df.at[idx, 'num_query_items']}")
                            else:
                                print("  WARNING: Could not calculate num_query_items due to missing values")

                            # Calculate effective_items for comparison
                            df.at[idx, "effective_items"] = df.at[idx, "num_query_items"]
                            print(f"  Set effective_items: {df.at[idx, 'effective_items']}")

                            # Calculate throughput if mean_latency_ms is available
                            if (
                                pd.notna(row["mean_latency_ms"])
                                and row["mean_latency_ms"] > 0
                                and pd.notna(df.at[idx, "num_query_items"])
                            ):
                                df.at[idx, "throughput_items_per_sec"] = df.at[idx, "num_query_items"] / (
                                    row["mean_latency_ms"] / 1000.0
                                )
                                print(
                                    f"  Calculated throughput: {df.at[idx, 'throughput_items_per_sec']:.2f} items/sec"
                                )

                            # Print all extracted params for debugging
                            print(
                                f"  Extracted SDPA model params for {row['model_config_name_raw']}: "
                                + f"batch_size={df.at[idx, 'batch_size']}, "
                                + f"num_q_heads={df.at[idx, 'num_q_heads']}, "
                                + f"num_kv_heads={df.at[idx, 'num_kv_heads']}, "
                                + f"head_dim={df.at[idx, 'head_dim']}, "
                                + f"seq_len={df.at[idx, 'seq_len']}"
                            )
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"  ERROR parsing model_params_raw for SDPA model_configs row {idx}: {e}")
                        print(f"  Raw value: {row['model_params_raw']}")

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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug output")

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(" PROXY ATTENTION LAB BENCHMARK ANALYZER ")
    print("=" * 80)

    # Find all JSON files in the results directory with the new naming pattern
    # Exclude results.json which is our output file
    json_files = [f for f in args.results_dir.glob("*.json") if f.name != "results.json"]

    if not json_files:
        print(f"No benchmark result files found in {args.results_dir}")
        return

    print(f"Found {len(json_files)} benchmark result files")

    # Group files by type for better debugging
    py_pal_files = [f for f in json_files if "py_pal_" in f.name or "pal_" in f.name]
    py_sdpa_files = [f for f in json_files if "py_sdpa_" in f.name or "sdpa_" in f.name]
    py_all_files = [f for f in json_files if "py_all_" in f.name]
    cpp_files = [f for f in json_files if "cpp_" in f.name]
    other_files = [f for f in json_files if f not in py_pal_files + py_sdpa_files + py_all_files + cpp_files]

    print(f"  - PAL Python files: {len(py_pal_files)}")
    print(f"  - SDPA Python files: {len(py_sdpa_files)}")
    print(f"  - Python combined files: {len(py_all_files)}")
    print(f"  - C++ files: {len(cpp_files)}")
    print(f"  - Other files: {len(other_files)}")

    if args.verbose:
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
    summary = (
        df[["benchmark_name_base", "source", "mean_latency_ms"]].groupby(["benchmark_name_base", "source"]).count()
    )
    print(summary)

    # Check which benchmark base names exist for each implementation
    print("\nAvailable benchmark types by source:")
    for source, group in df.groupby("source"):
        print(f"\n{source}:")
        benchmarks = sorted(group["benchmark_name_base"].unique())
        for bench in benchmarks:
            count = len(group[group["benchmark_name_base"] == bench])
            print(f"  - {bench} ({count} datapoints)")

    # Dictionary to track generated plot filenames
    plot_filenames = {}

    # Generate consolidated plots
    print("\n" + "=" * 80)
    print(" GENERATING PLOTS ")
    print("=" * 80)

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
    print("\n" + "=" * 80)
    print(" GENERATING JSON REPORT ")
    print("=" * 80)
    generate_json_report(df, args.output_dir, plot_filenames)

    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE ")
    print("=" * 80)
    print(f"All results saved to {args.output_dir}")
    if plot_filenames:
        print(f"✅ Generated plots: {', '.join(plot_filenames.values())}")
    else:
        print("⚠️ No plots were generated - check debug output for errors")
    print("✅ Generated JSON report: results.json")


if __name__ == "__main__":
    main()
