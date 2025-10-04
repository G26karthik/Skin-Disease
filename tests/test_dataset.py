"""Unit tests for dataset module."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.dataset import (
    SkinLesionDataset, get_train_transforms, get_val_transforms
)
from src.config import CLASS_NAMES, IMAGE_SIZE


def test_train_transforms():
    """Test training transforms."""
    transform = get_train_transforms()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Apply transform
    transformed = transform(image=dummy_image)
    
    # Check output
    assert 'image' in transformed
    assert isinstance(transformed['image'], torch.Tensor)
    assert transformed['image'].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert transformed['image'].dtype == torch.float32


def test_val_transforms():
    """Test validation transforms."""
    transform = get_val_transforms()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Apply transform
    transformed = transform(image=dummy_image)
    
    # Check output
    assert 'image' in transformed
    assert isinstance(transformed['image'], torch.Tensor)
    assert transformed['image'].shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_transform_normalization():
    """Test that transforms properly normalize images."""
    transform = get_val_transforms()
    
    # Create white image
    white_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    
    # Apply transform
    transformed = transform(image=white_image)
    image_tensor = transformed['image']
    
    # Check that values are normalized (should be around mean/std)
    # For white image (1.0) normalized: (1.0 - mean) / std
    assert image_tensor.min() < 3.0  # Reasonable range after normalization
    assert image_tensor.max() > -3.0


def test_class_names():
    """Test that class names are correctly defined."""
    assert len(CLASS_NAMES) == 3
    assert 'Benign' in CLASS_NAMES
    assert 'Suspicious' in CLASS_NAMES
    assert 'Urgent' in CLASS_NAMES


def test_image_size():
    """Test that image size is reasonable."""
    assert IMAGE_SIZE > 0
    assert IMAGE_SIZE == 224  # Standard for EfficientNet_B0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
