"""Base pipeline interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class Pipeline(ABC):
    """Abstract base class for model pipelines."""

    def __init__(self, model: Any, tokenizer: Optional[Any] = None):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        """Execute the pipeline on inputs.

        Args:
            inputs: Input data to process
            **kwargs: Additional pipeline parameters

        Returns:
            Pipeline output
        """
        pass

    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """Preprocess inputs for the model.

        Args:
            inputs: Raw input data

        Returns:
            Preprocessed inputs
        """
        pass

    @abstractmethod
    def postprocess(self, outputs: Any) -> Any:
        """Postprocess model outputs.

        Args:
            outputs: Raw model outputs

        Returns:
            Processed outputs
        """
        pass
