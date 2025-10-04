"""Unit tests for inference module."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.inference import (
    predict_from_tensor, predict_from_array, is_confident
)
from src.config import CLASS_NAMES, DEVICE


def test_predict_from_tensor():
    """Test prediction from tensor."""
    # Create dummy model
    model = Mock()
    model.eval = Mock()
    model.get_gradcam_target_layer = Mock()
    
    # Mock model output
    dummy_logits = torch.randn(1, 3)
    model.return_value = dummy_logits
    
    # Create dummy input
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    # Make prediction (without Grad-CAM to avoid complex mocking)
    result = predict_from_tensor(
        dummy_tensor,
        model=model,
        generate_gradcam=False
    )
    
    # Check result structure
    assert 'class_index' in result
    assert 'label' in result
    assert 'confidence' in result
    assert 'probabilities' in result
    assert 'class_probabilities' in result
    
    # Check types
    assert isinstance(result['class_index'], int)
    assert isinstance(result['label'], str)
    assert isinstance(result['confidence'], float)
    assert isinstance(result['probabilities'], np.ndarray)
    assert isinstance(result['class_probabilities'], dict)
    
    # Check values
    assert 0 <= result['class_index'] < 3
    assert result['label'] in CLASS_NAMES
    assert 0.0 <= result['confidence'] <= 1.0
    assert len(result['probabilities']) == 3
    assert len(result['class_probabilities']) == 3


def test_predict_from_array_shape():
    """Test that predict_from_array handles various input shapes."""
    # Create dummy model
    model = Mock()
    model.eval = Mock()
    model.get_gradcam_target_layer = Mock()
    
    # Mock model output
    model.return_value = torch.randn(1, 3)
    
    # Test with different image sizes
    for size in [(224, 224, 3), (300, 300, 3), (512, 512, 3)]:
        dummy_image = np.random.randint(0, 255, size, dtype=np.uint8)
        
        # This should not raise an error
        try:
            result = predict_from_array(
                dummy_image,
                model=model,
                generate_gradcam=False
            )
            # If we get here, the function handled the input correctly
            assert True
        except Exception as e:
            pytest.fail(f"predict_from_array failed with image size {size}: {e}")


def test_is_confident():
    """Test confidence threshold checking."""
    # High confidence
    result_high = {'confidence': 0.9}
    assert is_confident(result_high, threshold=0.7)
    assert is_confident(result_high, threshold=0.5)
    
    # Low confidence
    result_low = {'confidence': 0.4}
    assert not is_confident(result_low, threshold=0.7)
    assert not is_confident(result_low, threshold=0.5)
    
    # Edge case
    result_edge = {'confidence': 0.5}
    assert is_confident(result_edge, threshold=0.5)
    assert not is_confident(result_edge, threshold=0.51)


def test_result_probabilities_sum():
    """Test that probabilities sum to 1."""
    model = Mock()
    model.eval = Mock()
    model.get_gradcam_target_layer = Mock()
    
    # Create realistic softmax output
    logits = torch.tensor([[1.5, -0.5, 0.2]])
    model.return_value = logits
    
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    result = predict_from_tensor(
        dummy_tensor,
        model=model,
        generate_gradcam=False
    )
    
    # Check that probabilities sum to ~1.0
    prob_sum = sum(result['class_probabilities'].values())
    assert abs(prob_sum - 1.0) < 0.001, f"Probabilities sum to {prob_sum}, not 1.0"


def test_class_probabilities_dict():
    """Test that class_probabilities dict has correct structure."""
    model = Mock()
    model.eval = Mock()
    model.get_gradcam_target_layer = Mock()
    model.return_value = torch.randn(1, 3)
    
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    result = predict_from_tensor(
        dummy_tensor,
        model=model,
        generate_gradcam=False
    )
    
    # Check all class names present
    for class_name in CLASS_NAMES:
        assert class_name in result['class_probabilities']
        assert isinstance(result['class_probabilities'][class_name], float)
        assert 0.0 <= result['class_probabilities'][class_name] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
