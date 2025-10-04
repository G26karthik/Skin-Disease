"""
Configuration module for AI Skin Lesion Classifier.

This module centralizes all hyperparameters, paths, device settings,
and reproducibility configurations for the project.

Example usage:
    from src.config import DEVICE, BATCH_SIZE, LEARNING_RATE
    
    print(f"Training on: {DEVICE}")
    model = model.to(DEVICE)
"""

import os
import random
import torch
import numpy as np
from pathlib import Path
from typing import Tuple

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# Create directories if they don't exist
for directory in [MODELS_DIR, METRICS_DIR, RUNS_DIR, CHECKPOINT_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Hardware Configuration
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
USE_CUDA = torch.cuda.is_available()

# Enable cudnn benchmarking for speed on NVIDIA GPUs
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Benchmark mode for speed

# ============================================================================
# Data Configuration
# ============================================================================
IMAGE_SIZE = 224  # EfficientNet_B0 default input size
NUM_CLASSES = 3
CLASS_NAMES = ["Benign", "Suspicious", "Urgent"]

# ImageNet normalization (pretrained model requirement)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Data split ratios
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ============================================================================
# Training Hyperparameters
# ============================================================================
# Batch size with automatic adjustment for GPU memory
DEFAULT_BATCH_SIZE = 32
BATCH_SIZE = DEFAULT_BATCH_SIZE

# Adjust batch size for smaller GPUs (like 4050 with 6GB)
if USE_CUDA:
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory_gb < 8:
        BATCH_SIZE = 16
        print(f"GPU memory: {gpu_memory_gb:.1f}GB - Using batch size {BATCH_SIZE}")

NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3

# Learning rate scheduler
LR_SCHEDULER = "cosine"  # Options: "cosine", "plateau"
LR_PATIENCE = 3  # For ReduceLROnPlateau
LR_FACTOR = 0.5  # For ReduceLROnPlateau
LR_MIN = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 5

# ============================================================================
# DataLoader Configuration
# ============================================================================
# Use multiple workers on Linux/Mac, single worker on Windows to avoid issues
NUM_WORKERS = 0 if os.name == 'nt' else 4
PIN_MEMORY = USE_CUDA
PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
MODEL_PATH = MODELS_DIR / "model.pt"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint.pth"

# ============================================================================
# Mixed Precision Training
# ============================================================================
USE_AMP = USE_CUDA  # Automatic Mixed Precision for NVIDIA GPUs
AMP_DTYPE = torch.float16  # Use float16 for mixed precision

# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        
    Example:
        set_seed(42)
        # All random operations are now deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Note: Setting deterministic mode can slow down training
    # Uncomment if you need perfect reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ============================================================================
# Augmentation Configuration
# ============================================================================
AUGMENTATION_CONFIG = {
    "rotation_limit": 20,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "hue_shift_limit": 20,
    "saturation_shift_limit": 30,
    "flip_prob": 0.5,
    "crop_scale": (0.8, 1.0),
}

# ============================================================================
# Grad-CAM Configuration
# ============================================================================
GRADCAM_LAYER = "features"  # Layer name for Grad-CAM (EfficientNet final conv layer)
GRADCAM_ALPHA = 0.4  # Overlay transparency

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_INTERVAL = 10  # Log every N batches
SAVE_BEST_METRIC = "f1"  # Options: "f1", "accuracy", "loss"

# ============================================================================
# Inference Configuration
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for prediction
TTA_ENABLED = False  # Test-time augmentation

# ============================================================================
# Display Configuration
# ============================================================================
def print_config() -> None:
    """Print current configuration to console."""
    print("=" * 70)
    print("AI SKIN LESION CLASSIFIER - CONFIGURATION")
    print("=" * 70)
    print(f"Device: {DEVICE} ({GPU_NAME})")
    print(f"CUDA Available: {USE_CUDA}")
    print(f"Mixed Precision (AMP): {USE_AMP}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"Random Seed: {SEED}")
    print("=" * 70)


# Auto-set seed on import
set_seed(SEED)


if __name__ == "__main__":
    # Test configuration
    print_config()
    
    # Test seed setting
    set_seed(42)
    print(f"\nTest random values (should be reproducible):")
    print(f"Random float: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
