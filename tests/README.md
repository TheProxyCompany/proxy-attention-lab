# Tests

Each kernel in PAL has a dedicated subdirectory under `tests/`.

```
tests/
  paged_attention/
    unit_tests/        # functional tests
    benchmarks/
      python/         # pytest-benchmark scripts
      cpp/            # Google Benchmark executables
```

When adding a new kernel, create a similar subdirectory and populate it with unit tests and benchmarks.
