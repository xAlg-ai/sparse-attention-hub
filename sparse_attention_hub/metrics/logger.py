"""Singleton metric logger for micro metrics."""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import MicroMetric


class MicroMetricLogger:
    """Singleton logger for micro metrics."""

    _instance: Optional["MicroMetricLogger"] = None
    _initialized: bool = False

    def __new__(cls) -> "MicroMetricLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.available_metrics: List[MicroMetric] = []
            self.metrics_to_log: List[MicroMetric] = []
            self.path_to_log: str = "./metric_logs"
            self._ensure_log_directory()
            MicroMetricLogger._initialized = True

    def _ensure_log_directory(self) -> None:
        """Ensure the log directory exists."""
        os.makedirs(self.path_to_log, exist_ok=True)

    def register_metric(self, metric: MicroMetric) -> None:
        """Register a metric as available for logging.

        Args:
            metric: Metric instance to register
        """
        if metric not in self.available_metrics:
            self.available_metrics.append(metric)

    def should_log_metric(self, metric: MicroMetric) -> bool:
        """Check if a metric should be logged.

        Args:
            metric: Metric to check

        Returns:
            True if metric should be logged, False otherwise
        """
        return metric in self.metrics_to_log

    def enable_metric_logging(self, metric: MicroMetric) -> None:
        """Enable logging for a specific metric.

        Args:
            metric: Metric to enable logging for
        """
        if metric in self.available_metrics and metric not in self.metrics_to_log:
            self.metrics_to_log.append(metric)

    def disable_metric_logging(self, metric: MicroMetric) -> None:
        """Disable logging for a specific metric.

        Args:
            metric: Metric to disable logging for
        """
        if metric in self.metrics_to_log:
            self.metrics_to_log.remove(metric)

    def log(self, location: str, metric: MicroMetric, value: Any) -> None:
        """Log a metric value at a specific location.

        Args:
            location: Location identifier (e.g., layer name, step)
            metric: Metric that was computed
            value: Computed metric value
        """
        if not self.should_log_metric(metric):
            return

        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "location": location,
            "metric_name": metric.name,
            "metric_class": metric.__class__.__name__,
            "value": value,
        }

        # Write to log file
        log_filename = f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        log_path = os.path.join(self.path_to_log, log_filename)

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_available_metrics(self) -> List[MicroMetric]:
        """Get list of available metrics.

        Returns:
            List of available metric instances
        """
        return self.available_metrics.copy()

    def get_enabled_metrics(self) -> List[MicroMetric]:
        """Get list of metrics enabled for logging.

        Returns:
            List of enabled metric instances
        """
        return self.metrics_to_log.copy()

    def set_log_path(self, path: str) -> None:
        """Set the logging path.

        Args:
            path: New path for log files
        """
        self.path_to_log = path
        self._ensure_log_directory()

    def clear_logs(self) -> int:
        """Clear all log files.

        Returns:
            Number of log files deleted
        """
        count = 0
        for filename in os.listdir(self.path_to_log):
            if filename.startswith("metrics_") and filename.endswith(".jsonl"):
                file_path = os.path.join(self.path_to_log, filename)
                os.remove(file_path)
                count += 1
        return count
