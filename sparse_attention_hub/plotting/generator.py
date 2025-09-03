"""Plot generation for visualization."""

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .granularity import Granularity


class PlotGenerator:
    """Generates plots for sparse attention analysis.

    This class provides functionality to generate various types of plots
    for analyzing sparse attention patterns at different granularities.

    Attributes:
        storage_path: Directory path where generated plots will be saved.
    """

    def __init__(self, storage_path: str = "./plots") -> None:
        """Initialize the PlotGenerator.

        Args:
            storage_path: Directory path for storing generated plots.
                Defaults to "./plots".
        """
        self.storage_path = storage_path
        self._ensure_storage_directory()
        self._setup_plotting_style()

    def _setup_plotting_style(self) -> None:
        """Configure matplotlib and seaborn plotting styles."""
        plt.style.use("default")
        sns.set_palette("husl")

    def _ensure_storage_directory(self) -> None:
        """Ensure the storage directory exists."""
        os.makedirs(self.storage_path, exist_ok=True)

    def generate_plot(
        self,
        granularity: Granularity,
        data: Optional[Dict[str, Any]] = None,
        plot_type: str = "default",
        **kwargs: Any,
    ) -> str:
        """Generate a plot with specified granularity.

        Args:
            granularity: Level of granularity for the plot (per token, head, or layer).
            data: Optional data to plot. If None, generates sample data for demonstration.
            plot_type: Type of plot to generate. Currently unused but reserved for future extensions.
            **kwargs: Additional plotting parameters passed to the plot generation methods.

        Returns:
            Absolute path to the generated plot file.

        Raises:
            ValueError: If the specified granularity is not supported.
        """
        if granularity == Granularity.PER_TOKEN:
            return self._generate_line_plot(granularity, data, **kwargs)
        elif granularity == Granularity.PER_HEAD:
            return self._generate_heatmap_plot(granularity, data, **kwargs)
        elif granularity == Granularity.PER_LAYER:
            return self._generate_line_plot(granularity, data, **kwargs)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")

    def _generate_line_plot(
        self,
        granularity: Granularity,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate line plots for attention patterns.

        Args:
            granularity: Level of granularity (per token or per layer).
            data: Optional data to plot. If None, generates sample data.
            **kwargs: Additional plotting parameters.

        Returns:
            Path to the generated plot file.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if data is None:
            self._plot_sample_line_data(ax, granularity)
        else:
            self._plot_actual_line_data(ax, data)

        self._configure_line_plot_axes(ax, granularity)
        return self._save_plot(fig, "plot", granularity, data)

    def _plot_sample_line_data(self, ax: plt.Axes, granularity: Granularity) -> None:
        """Plot sample line data for demonstration.

        Args:
            ax: Matplotlib axes object to plot on.
            granularity: Granularity level for labeling.
        """
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        ax.plot(x, y, label=f"Sample {granularity.value} data")

    def _plot_actual_line_data(self, ax: plt.Axes, data: Dict[str, Any]) -> None:
        """Plot actual line data from provided data dictionary.

        Args:
            ax: Matplotlib axes object to plot on.
            data: Data dictionary containing plot information.
        """
        # Implementation depends on data structure
        # This is a placeholder for future actual data plotting
        pass

    def _configure_line_plot_axes(self, ax: plt.Axes, granularity: Granularity) -> None:
        """Configure axes for line plots.

        Args:
            ax: Matplotlib axes object to configure.
            granularity: Granularity level for title formatting.
        """
        title = (
            f"Sparse Attention Analysis - {granularity.value.replace('_', ' ').title()}"
        )
        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.set_ylabel("Attention Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _generate_heatmap_plot(
        self,
        granularity: Granularity,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate heatmaps for attention matrices.

        Args:
            granularity: Level of granularity (typically per head).
            data: Optional data containing attention matrix. If None, generates sample data.
            **kwargs: Additional plotting parameters.

        Returns:
            Path to the generated plot file.

        Raises:
            ValueError: If data is provided but doesn't contain required attention_matrix.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        attention_matrix = self._get_attention_matrix(granularity, data)
        self._create_heatmap(ax, attention_matrix)
        self._configure_heatmap_axes(ax, granularity)

        return self._save_plot(fig, "heatmap", granularity, data)

    def _get_attention_matrix(
        self, granularity: Granularity, data: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Get attention matrix from data or generate sample data.

        Args:
            granularity: Granularity level for determining matrix size.
            data: Optional data dictionary containing attention matrix.

        Returns:
            Attention matrix as numpy array.

        Raises:
            ValueError: If data is provided but doesn't contain attention_matrix.
        """
        if data is None:
            return self._generate_sample_attention_matrix(granularity)

        attention_matrix = data.get("attention_matrix")
        if attention_matrix is None:
            raise ValueError("attention_matrix required in data for heatmap")
        return attention_matrix

    def _generate_sample_attention_matrix(self, granularity: Granularity) -> np.ndarray:
        """Generate sample attention matrix for demonstration.

        Args:
            granularity: Granularity level for determining matrix size.

        Returns:
            Symmetric attention matrix.
        """
        size = 12 if granularity == Granularity.PER_HEAD else 8
        attention_matrix = np.random.rand(size, size)
        return (attention_matrix + attention_matrix.T) / 2  # Make symmetric

    def _create_heatmap(self, ax: plt.Axes, attention_matrix: np.ndarray) -> None:
        """Create heatmap visualization.

        Args:
            ax: Matplotlib axes object to plot on.
            attention_matrix: Attention matrix to visualize.
        """
        sns.heatmap(
            attention_matrix,
            annot=True,
            cmap="viridis",
            square=True,
            ax=ax,
            cbar_kws={"label": "Attention Weight"},
        )

    def _configure_heatmap_axes(self, ax: plt.Axes, granularity: Granularity) -> None:
        """Configure axes for heatmap plots.

        Args:
            ax: Matplotlib axes object to configure.
            granularity: Granularity level for title formatting.
        """
        title = f"Attention Matrix - {granularity.value.replace('_', ' ').title()}"
        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

    def _save_plot(
        self,
        fig: plt.Figure,
        plot_type: str,
        granularity: Granularity,
        data: Optional[Dict[str, Any]],
    ) -> str:
        """Save plot to file and close figure.

        Args:
            fig: Matplotlib figure object to save.
            plot_type: Type of plot for filename.
            granularity: Granularity level for filename.
            data: Data used for generating unique filename hash.

        Returns:
            Absolute path to the saved plot file.
        """
        filename = f"{plot_type}_{granularity.value}_{hash(str(data))}.png"
        filepath = os.path.join(self.storage_path, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return filepath

    def generate_comparison_plot(
        self, data_dict: Dict[str, Any], granularity: Granularity
    ) -> str:
        """Generate comparison plot for multiple datasets.

        Args:
            data_dict: Dictionary mapping dataset labels to their corresponding data.
            granularity: Level of granularity for the comparison plot.

        Returns:
            Path to the generated comparison plot file.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        self._plot_comparison_data(ax, data_dict)
        self._configure_comparison_axes(ax, granularity)

        return self._save_plot(fig, "comparison", granularity, data_dict)

    def _plot_comparison_data(self, ax: plt.Axes, data_dict: Dict[str, Any]) -> None:
        """Plot comparison data for multiple datasets.

        Args:
            ax: Matplotlib axes object to plot on.
            data_dict: Dictionary mapping labels to data.
        """
        for label, data in data_dict.items():
            # Plot each dataset - implementation depends on data structure
            # This is a placeholder for future actual data plotting
            pass

    def _configure_comparison_axes(
        self, ax: plt.Axes, granularity: Granularity
    ) -> None:
        """Configure axes for comparison plots.

        Args:
            ax: Matplotlib axes object to configure.
            granularity: Granularity level for title formatting.
        """
        title = f"Comparison - {granularity.value.replace('_', ' ').title()}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def set_storage_path(self, path: str) -> None:
        """Set the storage path for generated plots.

        Args:
            path: New directory path where plots will be saved.
        """
        self.storage_path = path
        self._ensure_storage_directory()
