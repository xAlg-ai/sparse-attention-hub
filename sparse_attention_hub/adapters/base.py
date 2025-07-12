"""Base classes and interfaces for sparse attention adapters."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Optional, Union

from ..sparse_attention.base import SparseAttention, SparseAttentionConfig


@dataclass
class Request:
    """Defines how the request for sparse attention is structured.

    Unlike usual request which is just a string, while working with sparse-attention,
    we have context which is preprocessed and KV-Cached using full attention and one or
    more questions which use this context.
    """

    context: str
    questions: Union[str, List[str]]


@dataclass
class RequestResponse:
    """Defines the response to a request. It contains response to each question."""

    responses: Union[str, List[str]]


class ModelHubAdapterInterface(ABC):
    """Defines the interface between external model hosting libraries,
    e.g. huggingface transformers and sparse-attention-hub project.
    """

    @abstractmethod
    def create_model(self, model_name: str) -> Any:
        """Creates a model given model_name.

        Args:
            model_name: Name of the model to create

        Returns:
            model: The created model instance
        """
        pass

    @abstractmethod
    def process_request(self, request: Request) -> RequestResponse:
        """Processes request.

        Args:
            request: The request to process

        Returns:
            response: The response to the request
        """
        pass


class SparseAttentionAdapterInterface(ABC):
    """Defines the wrappers around sparse-attention-hub to match the requirements
    of external libraries e.g. huggingface transformers.
    """

    @abstractmethod
    def get_custom_attention_function(
        self, sparse_attention: SparseAttention
    ) -> Callable:
        """Returns custom_attention_fn callable with the correct signature required for external library.

        Args:
            sparse_attention: The sparse attention instance

        Returns:
            custom_attention_fn: Callable with correct signature for external library
        """
        pass


class ModelAdapter(SparseAttentionAdapterInterface, ModelHubAdapterInterface, ABC):
    """Defines the Model Adapter layer between sparse-attention-hub project and external
    libraries. Adapter is defined for each (model, sparse_attention) instance and owns the two objects.
    It implements the functionality of SparseAttentionAdapterInterface, ModelHubAdapterInterface
    interfaces.
    """

    def __init__(
        self, sparse_attention_config: Optional[SparseAttentionConfig], model_name: str
    ) -> None:
        """Initialize model adapter.

        Args:
            sparse_attention_config: Configuration for sparse attention. If None, adapter runs in dense-only mode.
            model_name: Name of the model to use
        """
        self.sparse_attention_config = sparse_attention_config
        self.sparse_attention: Optional[SparseAttention] = (
            SparseAttention.create_from_config(sparse_attention_config)
            if sparse_attention_config is not None
            else None
        )
        self.model_name = model_name
        self.model: Any = self.create_model(model_name)

    @abstractmethod
    @contextmanager
    def enable_sparse_mode(self) -> Generator[None, None, None]:
        """Context manager to temporarily enable sparse attention mode.

        Notes:
            By default adapter is always in dense mode implying full attention is run.
            In order to run in sparse mode, use the context manager like:
            with ModelAdapterObject.enable_sparse_mode():
                <do something while using sparse attention>

        Yields:
            None
        """
        pass

    @abstractmethod
    @contextmanager
    def enable_dense_mode(self) -> Generator[None, None, None]:
        """Context manager to explicitly enable dense attention mode.

        Note: This is the default mode, so this context manager is mainly
        for clarity and consistency with enable_sparse_mode.

        Yields:
            None
        """
        pass
