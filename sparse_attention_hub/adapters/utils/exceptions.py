"""Exception classes for ModelServer operations."""


class ModelServerError(Exception):
    """Base exception class for ModelServer errors."""

    def __init__(self, message: str, details: str = "") -> None:
        """Initialize ModelServerError.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class ModelCreationError(ModelServerError):
    """Exception raised when model creation fails."""

    def __init__(
        self, model_name: str, gpu_id: int = None, original_error: Exception = None
    ) -> None:
        """Initialize ModelCreationError.

        Args:
            model_name: Name of the model that failed to create
            gpu_id: GPU ID where creation was attempted
            original_error: Original exception that caused the failure
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.original_error = original_error

        details = f"model_name={model_name}, gpu_id={gpu_id}"
        if original_error:
            details += f", original_error={str(original_error)}"

        super().__init__(f"Failed to create model: {model_name}", details)


class TokenizerCreationError(ModelServerError):
    """Exception raised when tokenizer creation fails."""

    def __init__(self, tokenizer_name: str, original_error: Exception = None) -> None:
        """Initialize TokenizerCreationError.

        Args:
            tokenizer_name: Name of the tokenizer that failed to create
            original_error: Original exception that caused the failure
        """
        self.tokenizer_name = tokenizer_name
        self.original_error = original_error

        details = f"tokenizer_name={tokenizer_name}"
        if original_error:
            details += f", original_error={str(original_error)}"

        super().__init__(f"Failed to create tokenizer: {tokenizer_name}", details)


class ReferenceCountError(ModelServerError):
    """Exception raised when reference counting operations fail."""

    def __init__(self, resource_key: str, operation: str, current_count: int) -> None:
        """Initialize ReferenceCountError.

        Args:
            resource_key: Key of the resource (model or tokenizer)
            operation: Operation that failed (increment, decrement, etc.)
            current_count: Current reference count
        """
        self.resource_key = resource_key
        self.operation = operation
        self.current_count = current_count

        details = f"resource_key={resource_key}, operation={operation}, current_count={current_count}"
        super().__init__(f"Reference count error during {operation}", details)


class ResourceCleanupError(ModelServerError):
    """Exception raised when resource cleanup fails."""

    def __init__(
        self, resource_key: str, resource_type: str, original_error: Exception = None
    ) -> None:
        """Initialize ResourceCleanupError.

        Args:
            resource_key: Key of the resource that failed to cleanup
            resource_type: Type of resource (model, tokenizer)
            original_error: Original exception that caused the failure
        """
        self.resource_key = resource_key
        self.resource_type = resource_type
        self.original_error = original_error

        details = f"resource_key={resource_key}, resource_type={resource_type}"
        if original_error:
            details += f", original_error={str(original_error)}"

        super().__init__(f"Failed to cleanup {resource_type}", details)
