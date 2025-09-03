"""MicroMetricLogger implementation for sparse attention hub."""

import inspect
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class LogEvent:
    """Log event data structure for metric logging.
    
    Attributes:
        timestamp: When the event was logged.
        metric: Metric identifier string.
        value: The actual metric value (can be None).
        metadata: Additional context information like layer, head, etc.
        location: Auto-inferred location as "module.function" or "class.method".
    """

    timestamp: datetime
    metric: str
    value: Union[None, Any]
    metadata: Dict[str, Any]
    location: str


class MicroMetricLogger:
    """Singleton logger for micro metrics with queue-based architecture.
    
    This class provides a singleton pattern for logging micro metrics during
    sparse attention operations. It supports metric registration, sampling,
    and configurable limits on record count and flush behavior.
    
    Attributes:
        log_path: Optional directory path where log files will be written.
        flush_every: Number of events after which to flush to disk.
        flush_interval: Time interval in seconds for automatic flushing.
        max_records: Maximum number of records to log (None for unlimited).
        sampling_factor: Probability of logging each event (0.0-1.0).
    """

    _instance: Optional["MicroMetricLogger"] = None
    _initialized: bool = False
    _registered_metrics: Dict[str, type] = {}

    def __new__(cls, *args, **kwargs) -> "MicroMetricLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_path: Optional[str] = None,
        flush_every: int = 1000,
        flush_interval: float = 60.0,
        enabled_metrics: Union[List[str], str, None] = None,
        max_records: Optional[int] = None,
        sampling_factor: float = 1.0,
    ) -> None:
        """Initialize the MicroMetricLogger.
        
        Args:
            log_path: Optional directory path where log files will be written.
            flush_every: Number of events after which to flush to disk.
            flush_interval: Time interval in seconds for automatic flushing.
            enabled_metrics: List of metric identifiers to enable, "all", or None.
            max_records: Maximum number of events to log (None for unlimited).
            sampling_factor: Probability of logging each event (0.0-1.0).
        """
        if not self._initialized:
            self._initialize_logger(
                log_path, flush_every, flush_interval, 
                enabled_metrics, max_records, sampling_factor
            )
        else:
            self._handle_reinitialization(log_path)

    def _initialize_logger(
        self,
        log_path: Optional[str],
        flush_every: int,
        flush_interval: float,
        enabled_metrics: Union[List[str], str, None],
        max_records: Optional[int],
        sampling_factor: float,
    ) -> None:
        """Initialize logger instance for the first time.
        
        Args:
            log_path: Directory path for log files.
            flush_every: Number of events before flushing.
            flush_interval: Time interval for flushing.
            enabled_metrics: Metrics to enable initially.
            max_records: Maximum records limit.
            sampling_factor: Event sampling probability.
        """
        self._set_configuration(log_path, flush_every, flush_interval, max_records, sampling_factor)
        self._initialize_state()
        self._configure_logging_if_needed(log_path, enabled_metrics)
        MicroMetricLogger._initialized = True

    def _set_configuration(
        self,
        log_path: Optional[str],
        flush_every: int,
        flush_interval: float,
        max_records: Optional[int],
        sampling_factor: float,
    ) -> None:
        """Set logger configuration parameters.
        
        Args:
            log_path: Directory path for log files.
            flush_every: Number of events before flushing.
            flush_interval: Time interval for flushing.
            max_records: Maximum records limit.
            sampling_factor: Event sampling probability.
        """
        self.log_path = log_path
        self.flush_every = flush_every
        self.flush_interval = flush_interval
        self.max_records = max_records
        self.sampling_factor = max(0.0, min(1.0, sampling_factor))

    def _initialize_state(self) -> None:
        """Initialize internal state variables."""
        self.log_queue: deque = deque(maxlen=10000)
        self.enabled_metrics: set = set()
        self.last_flush_time = time.time()
        self._total_records_logged: int = 0

    def _configure_logging_if_needed(
        self, 
        log_path: Optional[str], 
        enabled_metrics: Union[List[str], str, None]
    ) -> None:
        """Configure logging if log path is provided.
        
        Args:
            log_path: Directory path for log files.
            enabled_metrics: Metrics to enable initially.
        """
        if log_path is not None:
            self._ensure_log_directory()
            self.enable_metrics(enabled_metrics)

    def _handle_reinitialization(self, log_path: Optional[str]) -> None:
        """Handle warning when logger is reinitialized.
        
        Args:
            log_path: New log path being requested.
        """
        if self.log_path and log_path and self.log_path != log_path:
            print(
                f"Warning: MicroMetricLogger already initialized with log_path: {self.log_path}"
            )

    @classmethod
    def register_metric(cls, identifier: str, dtype: type) -> None:
        """Register a metric with its string identifier and expected data type.

        This works at class level and doesn't require logger initialization.

        Args:
            identifier: Unique string identifier for the metric.
            dtype: Expected data type for the metric values.
        """
        if identifier in cls._registered_metrics:
            print(f"Warning: Metric '{identifier}' is being re-registered")
        cls._registered_metrics[identifier] = dtype

    @classmethod
    def get_registered_metrics(cls) -> Dict[str, type]:
        """Get all registered metrics at class level.
        
        Returns:
            Dictionary mapping metric identifiers to their expected data types.
        """
        return cls._registered_metrics.copy()

    def _ensure_log_directory(self) -> None:
        """Ensure the log directory exists."""
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

    def _get_calling_location(self) -> str:
        """Get the calling location using inspect module.
        
        Returns:
            String representation of the calling location in format
            "module.class.method" or "module.function".
        """
        try:
            caller_frame = inspect.currentframe().f_back.f_back
            if caller_frame is None:
                return "unknown"

            module_name = self._get_module_name(caller_frame)
            function_name = caller_frame.f_code.co_name
            class_name = self._get_class_name(caller_frame)

            if class_name:
                return f"{module_name}.{class_name}.{function_name}"
            else:
                return f"{module_name}.{function_name}"
        except Exception:
            return "unknown"

    def _get_module_name(self, frame: Any) -> str:
        """Get module name from frame.
        
        Args:
            frame: Stack frame object.
            
        Returns:
            Module name or "unknown" if not available.
        """
        module = inspect.getmodule(frame)
        return module.__name__ if module else "unknown"

    def _get_class_name(self, frame: Any) -> Optional[str]:
        """Get class name from frame if it's a method call.
        
        Args:
            frame: Stack frame object.
            
        Returns:
            Class name if available, None otherwise.
        """
        if "self" in frame.f_locals:
            return frame.f_locals["self"].__class__.__name__
        return None

    def __del__(self) -> None:
        """Cleanup when logger is destroyed."""
        self.flush()

    def enable_metrics(self, metrics: Union[List[str], str, None] = None) -> None:
        """Enable logging for specific metrics.

        Args:
            metrics: List of metric identifiers to enable, "all" for all registered 
                    metrics, or None to disable all metrics.
        """
        if metrics == "all":
            self.enabled_metrics = set(self._registered_metrics.keys())
        elif isinstance(metrics, (list, set)):
            self._enable_selected_metrics(metrics)
        else:
            self.enabled_metrics = set()

    def _enable_selected_metrics(self, metrics: Union[List[str], set]) -> None:
        """Enable only the selected metrics that are registered.
        
        Args:
            metrics: List or set of metric identifiers to enable.
        """
        valid_metrics = set(metrics) & set(self._registered_metrics.keys())
        invalid_metrics = set(metrics) - set(self._registered_metrics.keys())
        
        if invalid_metrics:
            print(
                f"Warning: Attempting to enable unregistered metrics: {invalid_metrics}"
            )
        
        self.enabled_metrics = valid_metrics

    def log(self, identifier: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a metric value with optional metadata.

        Location is auto-inferred from the calling context. This only works if log_path is defined.

        Args:
            identifier: Unique string identifier for the metric.
            value: The metric value to log.
            metadata: Optional additional context information.
        """
        if not self._should_log_metric(identifier):
            return

        if not self._is_within_limits():
            return

        if not self._passes_sampling():
            return

        event = self._create_log_event(identifier, value, metadata)
        self._add_event_and_flush_if_needed(event)

    def _should_log_metric(self, identifier: str) -> bool:
        """Check if the metric should be logged based on configuration.
        
        Args:
            identifier: Metric identifier to check.
            
        Returns:
            True if the metric should be logged, False otherwise.
        """
        if self.log_path is None:
            print(
                f"Warning: Cannot log metric '{identifier}' - log_path not defined. "
                "Use configure_logging() first."
            )
            return False

        if identifier not in self.enabled_metrics:
            print(
                f"Warning: Attempting to log metric '{identifier}' which is not enabled"
            )
            return False

        return True

    def _is_within_limits(self) -> bool:
        """Check if logging is within configured limits.
        
        Returns:
            True if within limits, False if max records reached.
        """
        return not (
            self.max_records is not None
            and self._total_records_logged >= self.max_records
        )

    def _passes_sampling(self) -> bool:
        """Check if event passes sampling filter.
        
        Returns:
            True if event should be logged based on sampling factor.
        """
        return self.sampling_factor >= 1.0 or random.random() <= self.sampling_factor

    def _create_log_event(
        self, 
        identifier: str, 
        value: Any, 
        metadata: Optional[Dict[str, Any]]
    ) -> LogEvent:
        """Create a log event from the provided data.
        
        Args:
            identifier: Metric identifier.
            value: Metric value.
            metadata: Optional metadata.
            
        Returns:
            LogEvent instance ready for logging.
        """
        return LogEvent(
            timestamp=datetime.now(),
            metric=identifier,
            value=value,
            metadata=metadata or {},
            location=self._get_calling_location(),
        )

    def _add_event_and_flush_if_needed(self, event: LogEvent) -> None:
        """Add event to queue and flush if threshold is reached.
        
        Args:
            event: LogEvent to add to the queue.
        """
        self.log_queue.append(event)
        self._total_records_logged += 1

        if len(self.log_queue) >= self.flush_every:
            self.flush()

    def configure_logging(
        self,
        log_path: str,
        enabled_metrics: Union[List[str], str, None] = None,
        max_records: Optional[int] = None,
        sampling_factor: float = 1.0,
    ) -> None:
        """Configure logging with a log path and optionally enable metrics.

        This must be called before logging can work. Resets the total records counter.

        Args:
            log_path: Directory path where log files will be written.
            enabled_metrics: List of metric identifiers to enable, "all", or None.
            max_records: Maximum number of records to log (None for unlimited).
            sampling_factor: Probability of logging each event (0.0-1.0).
        """
        self.log_path = log_path
        self._ensure_log_directory()
        self.enable_metrics(enabled_metrics)
        self._update_logging_limits(max_records, sampling_factor)
        self._total_records_logged = 0

    def _update_logging_limits(self, max_records: Optional[int], sampling_factor: float) -> None:
        """Update logging limits and sampling factor.
        
        Args:
            max_records: Maximum records limit.
            sampling_factor: Event sampling probability.
        """
        self.max_records = max_records
        self.sampling_factor = max(0.0, min(1.0, sampling_factor))

    def flush(self) -> None:
        """Force flush the current queue to disk.
        
        Writes all queued events to the log file in JSONL format.
        """
        if not self.log_queue or self.log_path is None:
            return

        filepath = os.path.join(self.log_path, "micro_metrics.jsonl")
        self._write_events_to_file(filepath)
        self.last_flush_time = time.time()

    def _write_events_to_file(self, filepath: str) -> None:
        """Write all queued events to the specified file.
        
        Args:
            filepath: Path to the log file.
        """
        with open(filepath, "a", encoding="utf-8") as f:
            while self.log_queue:
                event = self.log_queue.popleft()
                event_dict = self._serialize_event(event)
                f.write(json.dumps(event_dict) + "\n")

    def _serialize_event(self, event: LogEvent) -> Dict[str, Any]:
        """Serialize a log event to a dictionary.
        
        Args:
            event: LogEvent to serialize.
            
        Returns:
            Dictionary representation of the event.
        """
        event_dict = asdict(event)
        event_dict["timestamp"] = event_dict["timestamp"].isoformat()
        return event_dict

    def is_metric_enabled(self, identifier: str) -> bool:
        """Check if a specific metric is enabled for logging.
        
        Args:
            identifier: Metric identifier to check.
            
        Returns:
            True if the metric is enabled, False otherwise.
        """
        return identifier in self.enabled_metrics

    def get_enabled_metrics(self) -> set:
        """Get currently enabled metrics.
        
        Returns:
            Set of enabled metric identifiers.
        """
        return self.enabled_metrics.copy()

    def is_logging_configured(self) -> bool:
        """Check if logging is configured.
        
        Returns:
            True if log_path is set, False otherwise.
        """
        return self.log_path is not None

    def get_total_records_logged(self) -> int:
        """Get the total number of records logged.
        
        Returns:
            Number of records logged since initialization or last configure_logging call.
        """
        return getattr(self, "_total_records_logged", 0)

    def is_max_records_reached(self) -> bool:
        """Check if the maximum number of records has been reached.
        
        Returns:
            True if max records limit has been reached, False otherwise.
        """
        if self.max_records is None:
            return False
        return self.get_total_records_logged() >= self.max_records

    def get_records_remaining(self) -> Optional[int]:
        """Get the number of records remaining before hitting max_records limit.

        Returns:
            Number of records remaining, or None if no limit is set.
        """
        if self.max_records is None:
            return None
        return max(0, self.max_records - self.get_total_records_logged())

    def get_sampling_factor(self) -> float:
        """Get the current sampling factor.
        
        Returns:
            Current sampling factor (0.0-1.0).
        """
        return getattr(self, "sampling_factor", 1.0)

    def get_max_records(self) -> Optional[int]:
        """Get the current max_records limit.
        
        Returns:
            Maximum records limit, or None if no limit is set.
        """
        return getattr(self, "max_records", None)
