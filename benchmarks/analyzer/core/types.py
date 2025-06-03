"""Common data types and interfaces for the analyzer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class BenchmarkData:
    """Container for processed benchmark data."""

    df: pd.DataFrame
    source_files: list[Path]
    kernel_types: dict[str, str]  # Maps benchmark names to kernel types
    metadata: dict[str, Any]


class PlotterInterface(ABC):
    """Abstract base class for all plotters."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the unique name of this plotter."""
        pass

    @abstractmethod
    def get_required_fields(self) -> list[str]:
        """Return list of required DataFrame columns."""
        pass

    @abstractmethod
    def can_plot(self, data: BenchmarkData) -> bool:
        """Check if this plotter can handle the given data."""
        pass

    @abstractmethod
    def plot(self, data: BenchmarkData, output_dir: Path, **kwargs) -> dict[str, Any]:
        """
        Generate plots and return results.

        Args:
            data: The benchmark data to plot
            output_dir: Directory to save plots
            **kwargs: Additional plotter-specific options

        Returns:
            Dictionary containing plot metadata and results
        """
        pass

    def get_description(self) -> str:
        """Return a description of what this plotter does."""
        return f"{self.get_name()} plotter"
