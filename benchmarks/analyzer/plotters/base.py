"""Base plotter implementation with common functionality."""

import logging
from pathlib import Path

from benchmarks.analyzer.core.types import BenchmarkData, PlotterInterface

logger = logging.getLogger(__name__)


class BasePlotter(PlotterInterface):
    """Base implementation for plotters with common functionality."""

    def can_plot(self, data: BenchmarkData) -> bool:
        """Default implementation checks if all required fields are present."""
        required_fields = self.get_required_fields()
        available_fields = set(data.df.columns)

        missing_fields = set(required_fields) - available_fields
        if missing_fields:
            logger.debug(f"{self.get_name()} missing fields: {missing_fields}")
            return False

        return True

    def ensure_output_dir(self, output_dir: Path) -> None:
        """Ensure the output directory exists."""
        output_dir.mkdir(parents=True, exist_ok=True)
