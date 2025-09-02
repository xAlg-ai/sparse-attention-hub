"""Unit tests for MicroMetricLogger."""

import json
import time
from datetime import datetime

import pytest

from sparse_attention_hub.metric_logging import LogEvent, MicroMetricLogger


@pytest.mark.unit
class TestLogEvent:
    """Test cases for LogEvent dataclass."""

    def test_log_event_creation(self) -> None:
        """Test LogEvent creation with all fields."""
        timestamp = datetime.now()
        event = LogEvent(
            timestamp=timestamp,
            metric="test_metric",
            value=42.0,
            metadata={"layer": 1, "head": 2},
            location="test.module.function",
        )

        assert event.timestamp == timestamp
        assert event.metric == "test_metric"
        assert event.value == 42.0
        assert event.metadata == {"layer": 1, "head": 2}
        assert event.location == "test.module.function"

    def test_log_event_with_none_value(self) -> None:
        """Test LogEvent creation with None value."""
        event = LogEvent(
            timestamp=datetime.now(),
            metric="test_metric",
            value=None,
            metadata={},
            location="test.module.function",
        )

        assert event.value is None

    def test_log_event_minimal_metadata(self) -> None:
        """Test LogEvent creation with minimal metadata."""
        event = LogEvent(
            timestamp=datetime.now(),
            metric="test_metric",
            value=42.0,
            metadata={},
            location="test.module.function",
        )

        assert event.metadata == {}


@pytest.mark.unit
class TestMicroMetricLoggerSingleton:
    """Test cases for MicroMetricLogger singleton pattern."""

    def test_singleton_behavior(self) -> None:
        """Test that MicroMetricLogger follows singleton pattern."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger1 = MicroMetricLogger()
        logger2 = MicroMetricLogger()

        assert logger1 is logger2
        assert id(logger1) == id(logger2)

    def test_singleton_initialization_once(self) -> None:
        """Test that initialization only happens once."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger1 = MicroMetricLogger(log_path="/tmp/test1")
        logger2 = MicroMetricLogger(log_path="/tmp/test2")

        # Second initialization should not change the log_path
        assert logger1.log_path == "/tmp/test1"
        assert logger2.log_path == "/tmp/test1"  # Should be the same as logger1


@pytest.mark.unit
class TestMicroMetricLoggerRegistration:
    """Test cases for metric registration functionality."""

    def test_register_metric(self) -> None:
        """Test metric registration."""
        # Reset registered metrics
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("attention_score", float)
        MicroMetricLogger.register_metric("sparsity_ratio", float)

        registered = MicroMetricLogger.get_registered_metrics()
        assert "attention_score" in registered
        assert "sparsity_ratio" in registered
        assert registered["attention_score"] == float
        assert registered["sparsity_ratio"] == float

    def test_register_metric_re_registration_warning(self, capsys) -> None:
        """Test warning when re-registering a metric."""
        # Reset registered metrics
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)
        MicroMetricLogger.register_metric(
            "test_metric", int
        )  # Re-register with different type

        captured = capsys.readouterr()
        assert "Warning: Metric 'test_metric' is being re-registered" in captured.out

    def test_get_registered_metrics_copy(self) -> None:
        """Test that get_registered_metrics returns a copy."""
        # Reset registered metrics
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)
        registered = MicroMetricLogger.get_registered_metrics()

        # Modify the returned dict - should not affect the original
        registered["new_metric"] = str

        original = MicroMetricLogger.get_registered_metrics()
        assert "new_metric" not in original


@pytest.mark.unit
class TestMicroMetricLoggerInitialization:
    """Test cases for MicroMetricLogger initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger = MicroMetricLogger()

        assert logger.log_path is None
        assert logger.flush_every == 1000
        assert logger.flush_interval == 60.0
        assert logger.enabled_metrics == set()
        assert len(logger.log_queue) == 0
        assert logger.log_queue.maxlen == 10000

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger = MicroMetricLogger(
            log_path="/tmp/test",
            flush_every=500,
            flush_interval=30.0,
            enabled_metrics=["metric1", "metric2"],
        )

        assert logger.log_path == "/tmp/test"
        assert logger.flush_every == 500
        assert logger.flush_interval == 30.0

    def test_initialization_with_log_path_creates_directory(self, tmp_path) -> None:
        """Test that initialization creates log directory when log_path is provided."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        log_dir = tmp_path / "logs"
        _ = MicroMetricLogger(log_path=str(log_dir))

        assert log_dir.exists()
        assert log_dir.is_dir()


@pytest.mark.unit
class TestMicroMetricLoggerEnableMetrics:
    """Test cases for enabling metrics functionality."""

    def test_enable_metrics_all(self) -> None:
        """Test enabling all registered metrics."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        # Register some metrics
        MicroMetricLogger.register_metric("metric1", float)
        MicroMetricLogger.register_metric("metric2", int)

        logger = MicroMetricLogger()
        logger.enable_metrics("all")

        assert logger.enabled_metrics == {"metric1", "metric2"}

    def test_enable_metrics_specific_list(self) -> None:
        """Test enabling specific metrics from a list."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        # Register some metrics
        MicroMetricLogger.register_metric("metric1", float)
        MicroMetricLogger.register_metric("metric2", int)
        MicroMetricLogger.register_metric("metric3", str)

        logger = MicroMetricLogger()
        logger.enable_metrics(["metric1", "metric3"])

        assert logger.enabled_metrics == {"metric1", "metric3"}

    def test_enable_metrics_with_unregistered_warning(self, capsys) -> None:
        """Test warning when enabling unregistered metrics."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        # Register only one metric
        MicroMetricLogger.register_metric("metric1", float)

        logger = MicroMetricLogger()
        logger.enable_metrics(["metric1", "unregistered_metric"])

        captured = capsys.readouterr()
        assert "Warning: Attempting to enable unregistered metrics" in captured.out
        assert logger.enabled_metrics == {"metric1"}

    def test_enable_metrics_none(self) -> None:
        """Test enabling no metrics (None)."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger = MicroMetricLogger()
        logger.enable_metrics(None)

        assert logger.enabled_metrics == set()

    def test_enable_metrics_empty_list(self) -> None:
        """Test enabling empty list of metrics."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger = MicroMetricLogger()
        logger.enable_metrics([])

        assert logger.enabled_metrics == set()


@pytest.mark.unit
class TestMicroMetricLoggerLogging:
    """Test cases for logging functionality."""

    def test_log_without_configuration_warning(self, capsys) -> None:
        """Test warning when logging without configuration."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger = MicroMetricLogger()
        logger.log("test_metric", 42.0)

        captured = capsys.readouterr()
        assert (
            "Warning: Cannot log metric 'test_metric' - log_path not defined"
            in captured.out
        )

    def test_log_disabled_metric_warning(self, capsys, tmp_path) -> None:
        """Test warning when logging disabled metric."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("metric1", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["metric1"])
        logger.log("metric2", 42.0)  # Log unenabled metric

        captured = capsys.readouterr()
        assert (
            "Warning: Attempting to log metric 'metric2' which is not enabled"
            in captured.out
        )

    def test_log_enabled_metric(self, tmp_path) -> None:
        """Test successful logging of enabled metric."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        logger.log("test_metric", 42.0, {"layer": 1, "head": 2})

        assert len(logger.log_queue) == 1
        event = logger.log_queue[0]
        assert event.metric == "test_metric"
        assert event.value == 42.0
        assert event.metadata == {"layer": 1, "head": 2}
        assert isinstance(event.timestamp, datetime)

    def test_log_with_none_metadata(self, tmp_path) -> None:
        """Test logging with None metadata (should use empty dict)."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        logger.log("test_metric", 42.0, None)

        assert len(logger.log_queue) == 1
        event = logger.log_queue[0]
        assert event.metadata == {}

    def test_log_auto_flush_on_queue_size(self, tmp_path) -> None:
        """Test automatic flushing when queue reaches flush_every size."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), flush_every=2)
        logger.enable_metrics(["test_metric"])

        # Log 2 events - should trigger flush
        logger.log("test_metric", 1.0)
        logger.log("test_metric", 2.0)

        # Queue should be empty after flush
        assert len(logger.log_queue) == 0


@pytest.mark.unit
class TestMicroMetricLoggerFlushing:
    """Test cases for flushing functionality."""

    def test_flush_empty_queue(self, tmp_path) -> None:
        """Test flushing empty queue."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.flush()  # Should not raise any error

        # No file should be created
        log_file = tmp_path / "micro_metrics.jsonl"
        assert not log_file.exists()

    def test_flush_without_log_path(self) -> None:
        """Test flushing without log_path configured."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger = MicroMetricLogger()
        logger.flush()  # Should not raise any error

    def test_flush_creates_file_and_writes_events(self, tmp_path) -> None:
        """Test that flush creates file and writes events correctly."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        # Add some events to queue
        logger.log("test_metric", 42.0, {"layer": 1})
        logger.log("test_metric", 43.0, {"layer": 2})

        # Flush
        logger.flush()

        # Check file was created
        log_file = tmp_path / "micro_metrics.jsonl"
        assert log_file.exists()

        # Check content
        with open(log_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Parse first line
        event1 = json.loads(lines[0])
        assert event1["metric"] == "test_metric"
        assert event1["value"] == 42.0
        assert event1["metadata"] == {"layer": 1}
        assert "timestamp" in event1
        assert "location" in event1

    def test_flush_clears_queue(self, tmp_path) -> None:
        """Test that flush clears the queue."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        # Add events to queue
        logger.log("test_metric", 42.0)
        logger.log("test_metric", 43.0)

        assert len(logger.log_queue) == 2

        # Flush
        logger.flush()

        # Queue should be empty
        assert len(logger.log_queue) == 0

    def test_flush_updates_last_flush_time(self, tmp_path) -> None:
        """Test that flush updates last_flush_time."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        initial_time = logger.last_flush_time

        # Wait a bit
        time.sleep(0.01)

        # Add event and flush
        logger.log("test_metric", 42.0)
        logger.flush()

        assert logger.last_flush_time > initial_time


@pytest.mark.unit
class TestMicroMetricLoggerUtilityMethods:
    """Test cases for utility methods."""

    def test_is_metric_enabled(self, tmp_path) -> None:
        """Test is_metric_enabled method."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("metric1", float)
        MicroMetricLogger.register_metric("metric2", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["metric1"])

        assert logger.is_metric_enabled("metric1") is True
        assert logger.is_metric_enabled("metric2") is False

    def test_get_enabled_metrics(self, tmp_path) -> None:
        """Test get_enabled_metrics method."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("metric1", float)
        MicroMetricLogger.register_metric("metric2", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["metric1", "metric2"])

        enabled = logger.get_enabled_metrics()
        assert enabled == {"metric1", "metric2"}

        # Should return a copy
        enabled.add("metric3")
        assert "metric3" not in logger.enabled_metrics

    def test_is_logging_configured(self) -> None:
        """Test is_logging_configured method."""
        # Reset singleton state
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger1 = MicroMetricLogger()
        assert logger1.is_logging_configured() is False

        # Reset singleton state again for second test
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger2 = MicroMetricLogger(log_path="/tmp/test")
        assert logger2.is_logging_configured() is True

    def test_configure_logging(self, tmp_path) -> None:
        """Test configure_logging method."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("metric1", float)

        logger = MicroMetricLogger()
        logger.configure_logging(str(tmp_path), ["metric1"])

        assert logger.log_path == str(tmp_path)
        assert logger.enabled_metrics == {"metric1"}
        assert tmp_path.exists()


@pytest.mark.unit
class TestMicroMetricLoggerLocationInference:
    """Test cases for automatic location inference."""

    def test_get_calling_location_function(self, tmp_path) -> None:
        """Test location inference for function calls."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        def test_function() -> None:
            logger.log("test_metric", 42.0)

        test_function()

        assert len(logger.log_queue) == 1
        event = logger.log_queue[0]
        assert "test_function" in event.location

    def test_get_calling_location_class_method(self, tmp_path) -> None:
        """Test location inference for class method calls."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        class TestClass:
            def test_method(self) -> None:
                logger.log("test_metric", 42.0)

        test_obj = TestClass()
        test_obj.test_method()

        assert len(logger.log_queue) == 1
        event = logger.log_queue[0]
        assert "TestClass" in event.location
        assert "test_method" in event.location


@pytest.mark.unit
class TestMicroMetricLoggerCleanup:
    """Test cases for cleanup functionality."""

    def test_del_flush(self, tmp_path) -> None:
        """Test that __del__ calls flush."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        # Add event to queue
        logger.log("test_metric", 42.0)

        # Manually call __del__ (simulating object destruction)
        logger.__del__()

        # Check that event was flushed to file
        log_file = tmp_path / "micro_metrics.jsonl"
        assert log_file.exists()

        with open(log_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1


@pytest.mark.unit
class TestMicroMetricLoggerMaxRecords:
    """Test cases for max_records functionality."""

    def test_max_records_initialization(self, tmp_path) -> None:
        """Test initialization with max_records parameter."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), max_records=5)
        logger.enable_metrics(["test_metric"])

        assert logger.get_max_records() == 5
        assert logger.get_total_records_logged() == 0
        assert not logger.is_max_records_reached()
        assert logger.get_records_remaining() == 5

    def test_max_records_limiting(self, tmp_path) -> None:
        """Test that max_records limits the number of events logged."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), max_records=3)
        logger.enable_metrics(["test_metric"])

        # Log 5 events, but only 3 should be recorded
        for i in range(5):
            logger.log("test_metric", float(i))

        assert logger.get_total_records_logged() == 3
        assert logger.is_max_records_reached()
        assert logger.get_records_remaining() == 0
        assert len(logger.log_queue) == 3

    def test_max_records_none_unlimited(self, tmp_path) -> None:
        """Test that max_records=None allows unlimited logging."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), max_records=None)
        logger.enable_metrics(["test_metric"])

        # Log many events
        for i in range(100):
            logger.log("test_metric", float(i))

        assert logger.get_total_records_logged() == 100
        assert not logger.is_max_records_reached()
        assert logger.get_records_remaining() is None
        assert logger.get_max_records() is None

    def test_max_records_configure_logging_reset(self, tmp_path) -> None:
        """Test that configure_logging resets the records counter."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), max_records=5)
        logger.enable_metrics(["test_metric"])

        # Log 3 events
        for i in range(3):
            logger.log("test_metric", float(i))

        assert logger.get_total_records_logged() == 3

        # Reconfigure logging - should reset counter
        logger.configure_logging(str(tmp_path), ["test_metric"], max_records=10)

        assert logger.get_total_records_logged() == 0
        assert logger.get_max_records() == 10
        assert logger.get_records_remaining() == 10

        # Log more events
        for i in range(5):
            logger.log("test_metric", float(i))

        assert logger.get_total_records_logged() == 5


@pytest.mark.unit
class TestMicroMetricLoggerSamplingFactor:
    """Test cases for sampling_factor functionality."""

    def test_sampling_factor_initialization(self, tmp_path) -> None:
        """Test initialization with sampling_factor parameter."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), sampling_factor=0.5)
        logger.enable_metrics(["test_metric"])

        assert logger.get_sampling_factor() == 0.5

    def test_sampling_factor_clamping(self, tmp_path) -> None:
        """Test that sampling_factor is clamped to [0.0, 1.0]."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        # Test values above 1.0
        logger = MicroMetricLogger(log_path=str(tmp_path), sampling_factor=1.5)
        assert logger.get_sampling_factor() == 1.0

        # Reset and test values below 0.0
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False

        logger = MicroMetricLogger(log_path=str(tmp_path), sampling_factor=-0.5)
        assert logger.get_sampling_factor() == 0.0

    def test_sampling_factor_zero_no_logging(self, tmp_path) -> None:
        """Test that sampling_factor=0.0 prevents all logging."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), sampling_factor=0.0)
        logger.enable_metrics(["test_metric"])

        # Try to log many events - none should be recorded
        for i in range(100):
            logger.log("test_metric", float(i))

        assert logger.get_total_records_logged() == 0
        assert len(logger.log_queue) == 0

    def test_sampling_factor_one_all_logging(self, tmp_path) -> None:
        """Test that sampling_factor=1.0 logs all events."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), sampling_factor=1.0)
        logger.enable_metrics(["test_metric"])

        # Log events - all should be recorded
        for i in range(10):
            logger.log("test_metric", float(i))

        assert logger.get_total_records_logged() == 10
        assert len(logger.log_queue) == 10

    def test_sampling_factor_partial_logging(self, tmp_path) -> None:
        """Test that sampling_factor reduces logging approximately as expected."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), sampling_factor=0.5)
        logger.enable_metrics(["test_metric"])

        # Log many events to get statistical significance
        # Set random seed for reproducible test
        import random

        random.seed(42)

        num_events = 1000
        for i in range(num_events):
            logger.log("test_metric", float(i))

        # With sampling_factor=0.5, we expect roughly half the events
        # Allow for some variance (±20% of expected)
        expected = num_events * 0.5
        recorded = logger.get_total_records_logged()
        tolerance = expected * 0.2

        assert expected - tolerance <= recorded <= expected + tolerance

    def test_sampling_factor_configure_logging_update(self, tmp_path) -> None:
        """Test that configure_logging can update sampling_factor."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path), sampling_factor=1.0)
        logger.enable_metrics(["test_metric"])

        assert logger.get_sampling_factor() == 1.0

        # Update sampling factor through configure_logging
        logger.configure_logging(str(tmp_path), ["test_metric"], sampling_factor=0.3)

        assert logger.get_sampling_factor() == 0.3


@pytest.mark.unit
class TestMicroMetricLoggerCombinedFeatures:
    """Test cases for max_records and sampling_factor working together."""

    def test_max_records_with_sampling(self, tmp_path) -> None:
        """Test that max_records and sampling_factor work together correctly."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        # Set sampling_factor=1.0 to ensure all events that pass other checks are logged
        logger = MicroMetricLogger(
            log_path=str(tmp_path), max_records=5, sampling_factor=1.0
        )
        logger.enable_metrics(["test_metric"])

        # Log 10 events, but only 5 should be recorded due to max_records
        for i in range(10):
            logger.log("test_metric", float(i))

        assert logger.get_total_records_logged() == 5
        assert logger.is_max_records_reached()
        assert len(logger.log_queue) == 5

    def test_sampling_with_max_records_zero_sampling(self, tmp_path) -> None:
        """Test max_records with sampling_factor=0.0."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(
            log_path=str(tmp_path), max_records=5, sampling_factor=0.0
        )
        logger.enable_metrics(["test_metric"])

        # Try to log events - none should be recorded due to sampling_factor=0.0
        for i in range(10):
            logger.log("test_metric", float(i))

        assert logger.get_total_records_logged() == 0
        assert (
            not logger.is_max_records_reached()
        )  # Never reached because nothing was logged
        assert len(logger.log_queue) == 0


@pytest.mark.unit
class TestMicroMetricLoggerEdgeCases:
    """Test cases for edge cases and error handling."""

    def test_circular_buffer_overflow(self, tmp_path) -> None:
        """Test that circular buffer prevents memory overflow."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        # Add more events than maxlen (10000)
        for i in range(10001):
            logger.log("test_metric", float(i))

        # Queue should not exceed maxlen
        assert len(logger.log_queue) <= 10000

    def test_log_with_complex_metadata(self, tmp_path) -> None:
        """Test logging with complex metadata structures."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        complex_metadata = {
            "nested": {"key": "value", "list": [1, 2, 3]},
            "array": [{"a": 1}, {"b": 2}],
            "boolean": True,
            "null": None,
        }

        logger.log("test_metric", 42.0, complex_metadata)
        logger.flush()

        # Check that complex metadata was serialized correctly
        log_file = tmp_path / "micro_metrics.jsonl"
        with open(log_file, "r") as f:
            event = json.loads(f.readline())

        assert event["metadata"] == complex_metadata

    def test_log_with_none_value(self, tmp_path) -> None:
        """Test logging with None value."""
        # Reset singleton state and registered metrics
        MicroMetricLogger._instance = None
        MicroMetricLogger._initialized = False
        MicroMetricLogger._registered_metrics.clear()

        MicroMetricLogger.register_metric("test_metric", float)

        logger = MicroMetricLogger(log_path=str(tmp_path))
        logger.enable_metrics(["test_metric"])

        logger.log("test_metric", None, {"note": "null value"})
        logger.flush()

        # Check that None value was handled correctly
        log_file = tmp_path / "micro_metrics.jsonl"
        with open(log_file, "r") as f:
            event = json.loads(f.readline())

        assert event["value"] is None
