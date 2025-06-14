"""Pytest configuration for PAL tests."""

import mlx.core as mx
import pytest


@pytest.fixture(autouse=True)
def clear_mlx_cache():
    """Automatically clear MLX cache before and after each test."""
    # Clear cache before test
    mx.clear_cache()

    # Run the test
    yield

    # Clear cache after test
    mx.clear_cache()


@pytest.fixture(autouse=True, scope="session")
def configure_mlx_memory():
    """Configure MLX memory settings for tests."""
    # Set a memory limit if needed (optional)
    # This can help catch memory issues earlier
    # mx.set_memory_limit(limit_gb=4)  # Uncomment if needed

    # Ensure we start with a clean state
    mx.clear_cache()

    yield

    # Final cleanup
    mx.clear_cache()
