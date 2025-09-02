"""Unit tests for ModelServer exceptions."""

import pytest

from sparse_attention_hub.adapters.utils.exceptions import (
    ModelCreationError,
    ModelServerError,
    ReferenceCountError,
    ResourceCleanupError,
    TokenizerCreationError,
)


@pytest.mark.unit
class TestModelServerError:
    """Test the base ModelServerError class."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = ModelServerError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == ""

    def test_error_with_details(self) -> None:
        """Test error creation with details."""
        error = ModelServerError("Test error", "Additional details")
        
        assert str(error) == "Test error: Additional details"
        assert error.message == "Test error"
        assert error.details == "Additional details"

    def test_error_inheritance(self) -> None:
        """Test that ModelServerError inherits from Exception."""
        error = ModelServerError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ModelServerError)


@pytest.mark.unit
class TestModelCreationError:
    """Test the ModelCreationError class."""

    def test_basic_model_creation_error(self) -> None:
        """Test basic model creation error."""
        error = ModelCreationError("test-model", 0)
        
        assert "Failed to create model: test-model" in str(error)
        assert error.model_name == "test-model"
        assert error.gpu_id == 0
        assert error.original_error is None

    def test_model_creation_error_with_cpu(self) -> None:
        """Test model creation error with CPU (None gpu_id)."""
        error = ModelCreationError("test-model", None)
        
        assert "Failed to create model: test-model" in str(error)
        assert error.model_name == "test-model"
        assert error.gpu_id is None
        assert error.original_error is None

    def test_model_creation_error_with_original_error(self) -> None:
        """Test model creation error with original exception."""
        original_error = ValueError("Original error message")
        error = ModelCreationError("test-model", 0, original_error)
        
        assert "Failed to create model: test-model" in str(error)
        assert error.model_name == "test-model"
        assert error.gpu_id == 0
        assert error.original_error == original_error
        assert "Original error message" in str(error)

    def test_model_creation_error_inheritance(self) -> None:
        """Test inheritance chain."""
        error = ModelCreationError("test-model", 0)
        
        assert isinstance(error, ModelServerError)
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestTokenizerCreationError:
    """Test the TokenizerCreationError class."""

    def test_basic_tokenizer_creation_error(self) -> None:
        """Test basic tokenizer creation error."""
        error = TokenizerCreationError("test-tokenizer")
        
        assert "Failed to create tokenizer: test-tokenizer" in str(error)
        assert error.tokenizer_name == "test-tokenizer"
        assert error.original_error is None

    def test_tokenizer_creation_error_with_original_error(self) -> None:
        """Test tokenizer creation error with original exception."""
        original_error = RuntimeError("Original tokenizer error")
        error = TokenizerCreationError("test-tokenizer", original_error)
        
        assert "Failed to create tokenizer: test-tokenizer" in str(error)
        assert error.tokenizer_name == "test-tokenizer"
        assert error.original_error == original_error
        assert "Original tokenizer error" in str(error)

    def test_tokenizer_creation_error_inheritance(self) -> None:
        """Test inheritance chain."""
        error = TokenizerCreationError("test-tokenizer")
        
        assert isinstance(error, ModelServerError)
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestReferenceCountError:
    """Test the ReferenceCountError class."""

    def test_reference_count_error(self) -> None:
        """Test reference count error creation."""
        error = ReferenceCountError("test_key", "decrement", -1)
        
        assert "Reference count error during decrement" in str(error)
        assert error.resource_key == "test_key"
        assert error.operation == "decrement"
        assert error.current_count == -1

    def test_reference_count_error_details(self) -> None:
        """Test that error includes all details."""
        error = ReferenceCountError("model_key_123", "increment", 5)
        
        error_str = str(error)
        assert "resource_key=model_key_123" in error_str
        assert "operation=increment" in error_str
        assert "current_count=5" in error_str

    def test_reference_count_error_inheritance(self) -> None:
        """Test inheritance chain."""
        error = ReferenceCountError("test_key", "decrement", -1)
        
        assert isinstance(error, ModelServerError)
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestResourceCleanupError:
    """Test the ResourceCleanupError class."""

    def test_basic_resource_cleanup_error(self) -> None:
        """Test basic resource cleanup error."""
        error = ResourceCleanupError("test_key", "model")
        
        assert "Failed to cleanup model" in str(error)
        assert error.resource_key == "test_key"
        assert error.resource_type == "model"
        assert error.original_error is None

    def test_resource_cleanup_error_with_original_error(self) -> None:
        """Test resource cleanup error with original exception."""
        original_error = OSError("Permission denied")
        error = ResourceCleanupError("tokenizer_key", "tokenizer", original_error)
        
        assert "Failed to cleanup tokenizer" in str(error)
        assert error.resource_key == "tokenizer_key"
        assert error.resource_type == "tokenizer"
        assert error.original_error == original_error
        assert "Permission denied" in str(error)

    def test_resource_cleanup_error_details(self) -> None:
        """Test that error includes all details."""
        error = ResourceCleanupError("model_key_456", "model")
        
        error_str = str(error)
        assert "resource_key=model_key_456" in error_str
        assert "resource_type=model" in error_str

    def test_resource_cleanup_error_inheritance(self) -> None:
        """Test inheritance chain."""
        error = ResourceCleanupError("test_key", "model")
        
        assert isinstance(error, ModelServerError)
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestExceptionChaining:
    """Test exception chaining and context."""

    def test_exception_chaining(self) -> None:
        """Test that exceptions can be properly chained."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            model_error = ModelCreationError("test-model", 0, e)
            
            assert model_error.original_error == e
            assert "Original error" in str(model_error)

    def test_exception_context_preservation(self) -> None:
        """Test that exception context is preserved."""
        original = RuntimeError("CUDA out of memory")
        cleanup_error = ResourceCleanupError("gpu_model", "model", original)
        
        # Should be able to access original exception details
        assert cleanup_error.original_error == original
        assert isinstance(cleanup_error.original_error, RuntimeError)
        assert "CUDA out of memory" in str(cleanup_error)

    def test_nested_exception_details(self) -> None:
        """Test detailed error information in nested exceptions."""
        original = FileNotFoundError("Model file not found")
        creation_error = ModelCreationError("missing-model", None, original)
        
        error_str = str(creation_error)
        
        # Should contain model creation details
        assert "Failed to create model: missing-model" in error_str
        assert "model_name=missing-model" in error_str
        assert "gpu_id=None" in error_str
        
        # Should contain original error details
        assert "Model file not found" in error_str
