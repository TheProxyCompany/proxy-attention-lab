# Benchmark Analyzer Guide

The Proxy Attention Lab (PAL) includes a comprehensive benchmarking and analysis system to evaluate the performance of paged attention kernels.

## Getting Started

This document explains how to run benchmarks and analyze the results using the provided tools.

## Running Benchmarks

The `benchmarks.sh` script provides a unified interface for running benchmarks:

```bash
# Run all benchmarks (C++ and Python)
./scripts/benchmarks.sh --run all

# Run only Python PAL benchmarks
./scripts/benchmarks.sh --run py_pal

# Run only C++ SDPA benchmarks
./scripts/benchmarks.sh --run cpp_sdpa

# Clear all previous benchmarks, run all benchmarks, and analyze
./scripts/benchmarks.sh --run --reset --analyze

# Only analyze existing benchmark results
./scripts/benchmarks.sh --analyze
```

## Benchmark Data Files

All benchmark results are stored as JSON files in the `.benchmarks/` directory with the following naming convention:

```
{test_type}_{benchmark_name}_{timestamp}.json
```

For example:
- `py_pal_test_pal_latency_vs_seq_len_20250517_101218.json`
- `cpp_all_BM_PAL_ModelConfig_20250517_103015.json`

## Analyzing Benchmarks

The benchmark analyzer is a modular package that processes and visualizes benchmark results:

```bash
# Analyze all benchmark results in .benchmarks/ directory
python scripts/analyze_benchmarks.py .benchmarks/ .benchmarks/

# Enable verbose output for debugging
python scripts/analyze_benchmarks.py .benchmarks/ .benchmarks/ --verbose
```

## Output Files

The analyzer generates:

1. Plot images in the specified output directory:
   - `latency_vs_seq_len.png` - Compares latency across different sequence lengths
   - `latency_vs_head_dim.png` - Compares latency across different head dimensions
   - `latency_vs_effective_items.png` - Compares latency across different query item counts
   - `model_configs_comparison.png` - Compares realistic model configurations (Llama3_70B_Sim, etc.)

2. A comprehensive `results.json` file containing:
   - Summary metrics for all benchmarks
   - Data points used to generate each plot
   - Plot file references

## Benchmark Types

PAL supports several types of benchmarks:

1. **Sequence Length Benchmarks**
   - Tests performance as sequence length varies
   - Python: `test_pal_latency_vs_seq_len`, `test_sdpa_latency_vs_seq_len`
   - C++: `BM_PAL_LatencyVsSeqLen`, `BM_SDPA_LatencyVsSeqLen`

2. **Head Dimension Benchmarks**
   - Tests performance as head dimension varies
   - Python: `test_pal_latency_vs_head_dim`, `test_sdpa_latency_vs_head_dim`
   - C++: `BM_PAL_LatencyVsHeadDim`, `BM_SDPA_LatencyVsHeadDim`

3. **Query Items Benchmarks**
   - Tests performance as number of query items varies
   - Python: `test_pal_latency_vs_query_items`, `test_sdpa_latency_vs_batch_size`
   - C++: `BM_PAL_LatencyVsNumItems`, `BM_SDPA_LatencyVsNumItems`

4. **Model Configuration Benchmarks**
   - Tests performance with realistic model configurations
   - Python: `test_pal_latency_model_configs`, `test_sdpa_latency_model_configs`
   - Includes various model configurations like Llama3_70B_Sim, Qwen_8B_Sim, etc.

## Creating Custom Visualizations

The benchmark analyzer can be extended to create custom visualizations:

1. Add a new Python module in `scripts/benchmark_analyzer/plotters/`
2. Implement a `plot` function that processes the DataFrame and generates visualizations
3. Update `analyze_benchmarks.py` to include your new plotter

## Troubleshooting

If benchmark analysis fails:

1. Check that the benchmark JSON files exist in the specified directory
2. Ensure the JSON files have the expected format
3. Run with `--verbose` flag to see detailed debug information
4. Verify that required dependencies (pandas, matplotlib) are installed
