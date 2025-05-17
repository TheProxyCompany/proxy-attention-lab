# Benchmarks

Each kernel in PAL has a dedicated subdirectory under `benchmarks/`.

```
benchmarks/
  paged_attention/  # Kernel name
    python/         # pytest-benchmark scripts
    cpp/            # Google Benchmark executables
```

When adding a new kernel, create a similar subdirectory and populate it with benchmarks.
