"""
Example demonstrating how to use ModelHubHF for attention registration.

This example shows how to:
1. Create sparse attention implementations
2. Register them with the HuggingFace transformers library
3. Configure models to use the registered attention functions
4. Use the models with custom attention mechanisms
"""

import torch
from typing import Optional

# Import sparse attention components
from sparse_attention_hub.sparse_attention import (
    ResearchAttention,
    ResearchAttentionConfig,
    EfficientAttention,
    EfficientAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMasker,
    LocalMaskerConfig,
    SinkMasker,
    SinkMaskerConfig,
)
from sparse_attention_hub.model_hub import ModelHubHF


def create_research_attention_with_local_and_sink():
    """Create a ResearchAttention with LocalMasker and SinkMasker."""
    # Configure maskers
    local_config = LocalMaskerConfig(window_size=8)
    sink_config = SinkMaskerConfig(sink_size=4)
    
    # Create ResearchAttention configuration
    research_config = ResearchAttentionConfig(
        masker_configs=[sink_config, local_config]
    )
    
    # Create the attention implementation
    return ResearchAttention.create_from_config(research_config)


def create_efficient_attention():
    """Create an EfficientAttention implementation."""
    efficient_config = EfficientAttentionConfig()
    return EfficientAttention(efficient_config)


def main():
    """Main example function."""
    print("🚀 ModelHubHF Attention Registration Example")
    print("=" * 50)
    
    # Initialize the ModelHubHF
    model_hub = ModelHubHF()
    print("✅ ModelHubHF initialized")
    
    # Create sparse attention implementations
    print("\n📦 Creating sparse attention implementations...")
    
    research_attention = create_research_attention_with_local_and_sink()
    efficient_attention = create_efficient_attention()
    
    print("✅ Created ResearchAttention with LocalMasker and SinkMasker")
    print("✅ Created EfficientAttention")
    
    # Register attention implementations
    print("\n🔧 Registering attention implementations...")
    
    try:
        research_name = model_hub.register_sparse_attention(
            research_attention,
            attention_name="local_sink_attention",
            config={"description": "Local attention with sink tokens"}
        )
        print(f"✅ Registered research attention as: {research_name}")
        
        efficient_name = model_hub.register_sparse_attention(
            efficient_attention,
            attention_name="efficient_sparse_attention",
            config={"description": "Efficient sparse attention implementation"}
        )
        print(f"✅ Registered efficient attention as: {efficient_name}")
        
    except ImportError as e:
        print(f"⚠️  Warning: {e}")
        print("💡 Install transformers library to enable full functionality:")
        print("   pip install transformers")
        return
    
    # List registered attention functions
    print("\n📋 Registered attention functions:")
    registered_functions = model_hub.list_registered_attention_functions()
    for name, adapter_class in registered_functions.items():
        print(f"  - {name}: {adapter_class}")
    
    # Demonstrate model configuration (mock example)
    print("\n🔧 Configuring model to use custom attention...")
    
    # Create a mock model for demonstration
    class MockModel:
        def __init__(self):
            self.config = MockConfig()
    
    class MockConfig:
        def __init__(self):
            self.attn_implementation = "eager"  # Default HF attention
    
    mock_model = MockModel()
    print(f"Model initial attention: {mock_model.config.attn_implementation}")
    
    # Configure model to use our registered attention
    model_hub.configure_model_attention(mock_model, research_name)
    print(f"Model configured attention: {mock_model.config.attn_implementation}")
    
    # Get adapter information
    print("\n🔍 Adapter information:")
    adapter = model_hub.get_attention_adapter(research_name)
    print(f"Adapter type: {type(adapter).__name__}")
    print(f"Adapter config: {adapter.config}")
    print(f"Underlying attention type: {type(adapter.sparse_attention).__name__}")
    print(f"Number of maskers: {len(adapter.sparse_attention.maskers)}")
    
    # Demonstrate attention forward pass (mock example)
    print("\n🧮 Demonstrating attention forward pass...")
    
    # Create sample tensors
    batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64
    queries = torch.randn(batch_size, num_heads, seq_len, head_dim)
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    print(f"Input shapes: Q={queries.shape}, K={keys.shape}, V={values.shape}")
    
    # Mock module for attention forward
    class MockAttentionModule:
        def __init__(self):
            self.training = False
    
    mock_module = MockAttentionModule()
    
    try:
        # Call the attention forward method
        output, weights = adapter.attention_forward(
            module=mock_module,
            query_states=queries,
            key_states=keys,
            value_states=values,
            scaling=1.0 / (head_dim ** 0.5),
            dropout=0.0
        )
        
        print(f"✅ Attention forward successful!")
        print(f"Output shape: {output.shape}")
        print(f"Weights shape: {weights.shape if weights is not None else 'None'}")
        
    except Exception as e:
        print(f"❌ Attention forward failed: {e}")
    
    # Cleanup: unregister attention functions
    print("\n🧹 Cleaning up...")
    model_hub.unregister_attention_function(research_name)
    model_hub.unregister_attention_function(efficient_name)
    
    remaining = model_hub.list_registered_attention_functions()
    print(f"Remaining registered functions: {len(remaining)}")
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Key takeaways:")
    print("  1. ModelHubHF allows registration of sparse attention implementations")
    print("  2. Registered attention can be used by HuggingFace models")
    print("  3. No need to replace forward methods - uses transformers' registry")
    print("  4. Supports different types of sparse attention (Research, Efficient)")
    print("  5. Provides lifecycle management (register/unregister)")


if __name__ == "__main__":
    main()