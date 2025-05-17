# Tests

Each kernel in PAL has a dedicated subdirectory under `tests/`.

```
tests/
  paged_attention/      # Kernel name
    test_core_functionality.py
    ... more tests
```

When adding a new kernel, create a similar subdirectory and populate it with unit tests.
