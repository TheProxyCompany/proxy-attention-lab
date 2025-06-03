# PAL Benchmarking System Guide

This guide explains the PAL (Proxy Attention Lab) benchmarking system architecture and how to add new benchmarks.

## System Architecture

The PAL benchmarking system provides a comprehensive framework for measuring and visualizing performance metrics across different attention kernel implementations. It consists of three integrated components:

1. **C++ Benchmarks** - High-precision performance measurements using Google Benchmark
2. **Data Loader** - Automated parameter extraction and data processing pipeline
3. **Visualization System** - Matplotlib-based plotters that generate publication-ready figures and JSON results

## Adding a New Benchmark Type

### Step 1: Create the C++ Benchmark

#### Naming Convention
Benchmark names should clearly indicate what is being measured and follow this pattern:
```
BM_<implementation>_<metric>Vs<parameter>
```

#### Examples
```cpp
// Prefill operation benchmarks
BM_PAL_PrefillLatencyVsSeqLen         // PAL prefill latency vs sequence length
BM_MLX_SDPA_PrefillLatencyVsSeqLen    // MLX SDPA prefill latency vs sequence length

// Decode operation benchmarks
BM_PAL_DecodeLatencyVsHistoryLen      // PAL decode latency vs history length
BM_MLX_SDPA_DecodeLatencyVsHistoryLen // MLX decode latency vs history length

// Batch operation benchmarks
BM_PAL_BatchLatencyVsNumSequences     // Batch latency vs number of sequences
BM_PAL_BatchLatencyVsHistoryLength    // Batch latency vs history length
```

#### Implementation Template
```cpp
static void BM_PAL_YourMetricVsParameter(benchmark::State& state) {
    // Extract benchmark parameters
    int param_value = state.range(0);

    // Setup test data
    // ... initialize arrays, configurations, etc.

    // Main benchmark loop
    for (auto _ : state) {
        // Perform the operation being benchmarked
        mx::array result = your_operation(...);
        result.eval();  // Ensure GPU computation completes
    }
}

// Register the benchmark
BENCHMARK(BM_PAL_YourMetricVsParameter)
    ->Arg(64)->Iterations(20)->Repetitions(10)
    ->Arg(256)->Iterations(20)->Repetitions(10)
    ->Arg(1024)->Iterations(20)->Repetitions(10);
```

### Step 2: Update the Data Loader

The data loader automatically extracts parameters from benchmark names. Add your pattern to `analyzer/core/data_loader.py`:

```python
# In the _process_google_benchmark method, find the parameter extraction section:
if "PrefillLatencyVsSeqLen" in name:
    row["sequence_length"] = param_value
elif "YourMetricVsParameter" in name:
    row["your_parameter_name"] = param_value
    # Handle multiple parameters if needed:
    if len(parts) >= 3 and parts[2].isdigit():
        row["second_parameter"] = int(parts[2])
```

The data loader will automatically convert nanosecond timings to milliseconds and handle statistical aggregation.

### Step 3: Create a Visualization Plotter

Create a new plotter in `analyzer/plotters/your_metric.py`:

```python
from benchmarks.analyzer.core import BenchmarkData, plot_styles, register_plotter
from benchmarks.analyzer.plotters.base import BasePlotter

@register_plotter
class YourMetricPlotter(BasePlotter):
    """Plotter for your metric visualization."""

    def get_name(self) -> str:
        return "your_metric_name"

    def get_required_fields(self) -> list[str]:
        return ["your_parameter_name", "mean_latency", "group", "name"]

    def plot(self, data: BenchmarkData, output_dir: Path, **kwargs) -> dict[str, Any]:
        # 1. Filter and process data
        df = data.df
        relevant_benchmarks = df[df["name"].str.contains("YourMetric")]

        # 2. Create visualizations (support 2x2 grids for multiple plots)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 3. Plot data with proper styling
        for group, group_data in relevant_benchmarks.groupby("group"):
            sorted_data = group_data.sort_values("your_parameter_name")
            ax.plot(sorted_data["your_parameter_name"],
                   sorted_data["mean_latency"],
                   marker='o', label=group)

        # 4. Apply consistent styling
        plot_styles.apply_common_plot_aesthetics(
            ax, title="Your Metric vs Parameter",
            xlabel="Parameter", ylabel="Latency (ms)",
            styles=STYLES, x_scale="log", y_scale="log"
        )

        # 5. Save figure and results
        fig.savefig(output_dir / "your_metric.png", dpi=300)

        # 6. Export data to JSON
        results = {"your_metric": {}}
        # ... populate results dictionary

        return {
            "filename": "your_metric.png",
            "benchmark_type": "your_metric"
        }
```

## Running Benchmarks

### Basic Commands

```bash
# Run all benchmarks with analysis
./scripts/benchmarks.sh --run --analyze

# Run only C++ benchmarks
./scripts/benchmarks.sh --run cpp --analyze

# Run only Python benchmarks
./scripts/benchmarks.sh --run py --analyze

# Analyze existing results
./scripts/benchmarks.sh --analyze
```

### Using Filters

The benchmark script supports regex filtering for precise control:

```bash
# Run specific benchmark patterns
./scripts/benchmarks.sh --run cpp "PrefillLatency" --analyze
./scripts/benchmarks.sh --run cpp "DecodeLatency" --analyze
./scripts/benchmarks.sh --run cpp "BatchLatency" --analyze

# Run both prefill and decode for complete analysis
./scripts/benchmarks.sh --run cpp "(PrefillLatencyVsSeqLen|DecodeLatencyVsHistoryLen)" --analyze

# Run benchmarks for specific implementations
./scripts/benchmarks.sh --run cpp "BM_PAL_" --analyze      # Only PAL benchmarks
./scripts/benchmarks.sh --run cpp "BM_MLX_" --analyze      # Only MLX benchmarks

# Run benchmarks with specific parameters
./scripts/benchmarks.sh --run cpp ".*\/1024\/" --analyze   # Only 1024 parameter
```

### Advanced Options

```bash
# Clear previous results before running
./scripts/benchmarks.sh --reset --run cpp --analyze

# Rebuild project without running benchmarks
./scripts/benchmarks.sh --rebuild-only

# Run specific Python benchmark modules
./scripts/benchmarks.sh --run py paged_attention
```

## Output Structure

### Directory Layout
```
.benchmarks/
├── cpp_<filter>_paged_attention_benchmarks.json  # Raw benchmark data
├── latency_vs_seq_len.png                        # Generated plots
├── batch_decode_latency.png
└── results.json                                  # Processed results
```

### Results JSON Structure

Results are organized hierarchically by benchmark type and sub-categories:

```json
{
  "latency_vs_seq_len": {
    "prefill": {
      "cpp_pal": {
        "64.0": 0.6845,     // sequence_length: latency_ms
        "256.0": 1.6249,
        "512.0": 3.8690
      },
      "cpp_mlx": { ... }
    },
    "decode": {
      "cpp_pal": {
        "64.0": 0.2927,     // history_length: latency_ms
        "128.0": 0.3297,
        "256.0": 0.4076
      },
      "cpp_mlx": { ... }
    }
  },
  "batch_decode_latency": {
    "latency_vs_num_sequences": {
      "H=256": {
        "1": 0.3978,        // num_sequences: latency_ms
        "2": 0.4123,
        "4": 0.4567
      }
    },
    "throughput_vs_num_sequences": {
      "H=256": {
        "1": 2513.2,        // num_sequences: tokens/sec
        "2": 4852.1,
        "4": 8756.3
      }
    }
  }
}
```

## Best Practices

1. **Benchmark Naming**: Use descriptive names that clearly indicate what is being measured
2. **Parameter Ranges**: Choose parameter values that span realistic use cases
3. **Iterations/Repetitions**: Use sufficient iterations (20) and repetitions (10) for statistical reliability
4. **Data Processing**: Let the framework handle timing conversions and statistical aggregation
5. **Visualization**: Create 2x2 grids when comparing multiple related metrics
6. **JSON Export**: Structure results logically for easy programmatic access

## Troubleshooting

- **Build Failures**: Run `./scripts/benchmarks.sh --rebuild-only` to rebuild without running benchmarks
- **Missing Results**: Check that benchmark names match patterns in the data loader
- **Visualization Issues**: Ensure all required fields are present in the DataFrame
- **Performance**: Use `SPDLOG_LEVEL=debug` for detailed logging during development


# Benchmark Filter Patterns

This document lists the available C++ benchmark patterns that can be used with the `--run cpp` filter option.

## Usage

```bash
./scripts/benchmarks.sh --run cpp "PATTERN" --analyze
```

## Available Benchmark Patterns

### All Benchmarks
- No filter (runs everything)

### Prefill + Decode for Complete Analysis
- `"(PrefillLatencyVsSeqLen|DecodeLatencyVsHistoryLen)"` - Both prefill and decode for complete latency analysis

### Prefill Benchmarks (Two-Pass)
- `"PrefillLatencyVsSeqLen"` - All prefill latency vs sequence length benchmarks (PAL and MLX)
- `"PAL_PrefillLatencyVsSeqLen"` - Only PAL prefill benchmarks
- `"MLX_SDPA_PrefillLatencyVsSeqLen"` - Only MLX SDPA prefill benchmarks

### Decode Benchmarks (Fused Kernel)
- `"DecodeLatencyVsHistoryLen"` - All decode history length benchmarks
- `"PAL_DecodeLatencyVsHistoryLen"` - Only PAL decode benchmarks
- `"MLX_SDPA_DecodeLatencyVsHistoryLen"` - Only MLX SDPA decode benchmarks

### Batch Decode Benchmarks (Fused Kernel)
- `"BatchLatency"` - All batch latency benchmarks
- `"BatchLatencyVsNumSequences"` - Batch latency varying number of sequences
- `"BatchLatencyVsHistoryLength"` - Batch latency varying history length

## Regex Examples

The filter accepts any valid regex pattern:

```bash
# Run only PAL benchmarks
./scripts/benchmarks.sh --run cpp "BM_PAL_.*"

# Run only decode-related benchmarks
./scripts/benchmarks.sh --run cpp ".*Decode.*"

# Run benchmarks with history length 1024
./scripts/benchmarks.sh --run cpp ".*/1024/.*"

# Run batch benchmarks with 32 sequences
./scripts/benchmarks.sh --run cpp ".*BatchLatency.*/32/.*"
```
