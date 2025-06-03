"""Registry system for dynamically discovering and managing plotters."""

import importlib
import inspect
import logging
from pathlib import Path

from benchmarks.analyzer.core.types import BenchmarkData, PlotterInterface

logger = logging.getLogger(__name__)


class PlotterRegistry:
    """Registry for managing plotter instances."""

    def __init__(self):
        self._plotters: dict[str, PlotterInterface] = {}
        self._auto_discover = True

    def register(self, plotter: PlotterInterface) -> None:
        """Register a plotter instance."""
        name = plotter.get_name()
        if name in self._plotters:
            logger.warning(f"Overwriting existing plotter: {name}")
        self._plotters[name] = plotter
        logger.info(f"Registered plotter: {name}")

    def get(self, name: str) -> PlotterInterface | None:
        """Get a plotter by name."""
        return self._plotters.get(name)

    def list_plotters(self) -> list[str]:
        """List all registered plotter names."""
        return list(self._plotters.keys())

    def find_compatible_plotters(self, data: BenchmarkData) -> list[PlotterInterface]:
        """Find all plotters that can handle the given data."""
        compatible = []
        for plotter in self._plotters.values():
            try:
                if plotter.can_plot(data):
                    compatible.append(plotter)
            except Exception as e:
                logger.error(f"Error checking plotter {plotter.get_name()}: {e}")
        return compatible

    def auto_discover(self, plotters_dir: Path) -> None:
        """Automatically discover and register plotters from a directory."""
        if not plotters_dir.exists():
            logger.warning(f"Plotters directory does not exist: {plotters_dir}")
            return

        # Import all Python files in the plotters directory
        for file_path in plotters_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            module_name = f"benchmarks.analyzer.plotters.{file_path.stem}"
            try:
                module = importlib.import_module(module_name)

                # Find all PlotterInterface subclasses in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, PlotterInterface) and obj is not PlotterInterface:
                        try:
                            instance = obj()
                            self.register(instance)
                        except Exception as e:
                            logger.error(f"Failed to instantiate plotter {name}: {e}")

            except Exception as e:
                logger.error(f"Failed to import module {module_name}: {e}")


# Global registry instance
_registry = PlotterRegistry()


def get_registry() -> PlotterRegistry:
    """Get the global plotter registry."""
    return _registry


def register_plotter(plotter_class: type[PlotterInterface]):
    """Decorator to automatically register a plotter class."""

    def wrapper(cls):
        try:
            instance = cls()
            get_registry().register(instance)
        except Exception as e:
            logger.error(f"Failed to register plotter {cls.__name__}: {e}")
        return cls

    return wrapper(plotter_class)
