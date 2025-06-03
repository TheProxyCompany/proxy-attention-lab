# Adding New Benchmarks

This guide explains how to add new benchmark types to the PAL benchmarking system.

## Overview

The benchmarking system consists of three components:
1. **C++ Benchmarks** - Measure performance using Google Benchmark
2. **Data Loader** - Extracts parameters and metrics from benchmark results
3. **Plotters** - Visualize results and save to JSON

## Adding a New Benchmark Type

### 1. Create the C++ Benchmark

Name your benchmark descriptively to indicate what's being measured:
```cpp
// Good naming patterns:
BM_PAL_LatencyVsSeqLen/64/...         // Measures latency vs sequence length
BM_PAL_ThroughputVsBatchSize/32/...   // Measures throughput vs batch size
BM_PAL_LatencyVsHeadDim/128/...       // Measures latency vs head dimension
```

The format is: `BM_<impl>_<metric>Vs<parameter>/<value>/...`

### 2. Update the Data Loader

Edit `analyzer/core/data_loader.py` in the parameter extraction section:

```python
# Find this section and add your pattern:
elif "ThroughputVsBatchSize" in name:
    row["batch_size"] = param_value
elif "LatencyVsHeadDim" in name:
    row["head_dim"] = param_value
# Add your new pattern here
```

### 3. Create a Plotter

Create a new file in `analyzer/plotters/` that:
- Inherits from `BasePlotter`
- Uses the `@register_plotter` decorator
- Implements required methods:
  - `get_name()` - Returns unique plotter name
  - `get_required_fields()` - Lists required DataFrame columns
  - `plot()` - Creates visualization and saves results

The plotter should:
- Calculate any derived metrics (e.g., throughput from latency)
- Create matplotlib visualizations
- Save results under its own key in `results.json`

## Running Benchmarks

```bash
# Run all benchmarks and analyze
./scripts/benchmarks.sh --run cpp --analyze

# Run specific kernel benchmarks
./scripts/benchmarks.sh --run cpp paged_attention --analyze

# Analyze existing results
./scripts/benchmarks.sh --analyze
```

## Results Structure

Results are saved to `.benchmarks/results.json` organized by benchmark type:

```json
{
  "latency_vs_seq_len": {
    "prefill": { ... },
    "decode": { ... }
  },
  "throughput_vs_batch_size": {
    "pal": { ... },
    "mlx": { ... }
  }
}
```

Each plotter determines its own structure based on what makes sense for that metric.
