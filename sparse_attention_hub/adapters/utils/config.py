"""Configuration classes for ModelServer."""

from dataclasses import dataclass


@dataclass
class ModelServerConfig:
    """Configuration for ModelServer behavior.

    Attributes:
        delete_on_zero_reference: If True, models/tokenizers are deleted immediately when reference count reaches 0.
                                 If False, they remain in memory until explicit cleanup.
        enable_stats_logging: Whether to enable detailed statistics logging.
    """

    delete_on_zero_reference: bool = False  # Lazy deletion by default
    enable_stats_logging: bool = True
