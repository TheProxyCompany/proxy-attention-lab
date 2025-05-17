# Tests

Each kernel in PAL has a dedicated subdirectory under `tests/`.

```
tests/
  paged_attention/ # functional tests, used for TDD
    test_core_functionality.py
```

When adding a new kernel, create a similar subdirectory and populate it with unit tests and benchmarks.
