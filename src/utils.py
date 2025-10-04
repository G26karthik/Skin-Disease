"""
Utility functions for the Skin Lesion Classifier.

Provides helper functions for image processing, visualization,
seed setting, and other common operations.

Example usage:
    from src.utils import load_image, denormalize_image, visualize_batch
    
    image = load_image("path/to/image.jpg")
    batch_viz = visualize_batch(images, labels, predictions)
"""

import random
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    CLASS_NAMES, DATA_DIR, SEED
)


def set_global_seed(seed: int = SEED) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        
    Example:
        set_global_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
) -> np.ndarray:
    """
    Load and resize an image.
    
    Args:
        image_path: Path to image file
        target_size: Target size (H, W)
        
    Returns:
        Image as numpy array (H, W, 3) in RGB format
        
    Example:
        image = load_image("image.jpg", target_size=(224, 224))
    """
    image = Image.open(image_path).convert('RGB')
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]))
    return np.array(image)


def save_image(image: np.ndarray, save_path: Union[str, Path]) -> None:
    """
    Save image to disk.
    
    Args:
        image: Image as numpy array
        save_path: Path to save image
        
    Example:
        save_image(image_array, "output.jpg")
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    Image.fromarray(image).save(save_path)
    print(f"Image saved to {save_path}")


def denormalize_image(
    image: Union[torch.Tensor, np.ndarray],
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> np.ndarray:
    """
    Denormalize an image tensor to [0, 255] uint8 format.
    
    Args:
        image: Normalized image (C, H, W) or (H, W, C)
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image as numpy array (H, W, C) in [0, 255]
        
    Example:
        original_image = denormalize_image(normalized_tensor)
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Convert (C, H, W) to (H, W, C) if needed
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    
    # Clip to [0, 1] and convert to uint8
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    return image


def visualize_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    class_names: List[str] = CLASS_NAMES,
    num_images: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a batch of images with labels and predictions.
    
    Args:
        images: Batch of images (N, C, H, W)
        labels: True labels (N,)
        predictions: Predicted labels (N,) [optional]
        class_names: List of class names
        num_images: Number of images to display
        save_path: Path to save visualization
        
    Example:
        visualize_batch(images, labels, predictions, num_images=16)
    """
    num_images = min(num_images, len(images))
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx in range(num_images):
        ax = axes[idx]
        
        # Denormalize and prepare image
        image = denormalize_image(images[idx])
        
        # Plot image
        ax.imshow(image)
        ax.axis('off')
        
        # Create title
        true_label = class_names[labels[idx]]
        
        if predictions is not None:
            pred_label = class_names[predictions[idx]]
            color = 'green' if labels[idx] == predictions[idx] else 'red'
            title = f"True: {true_label}\nPred: {pred_label}"
        else:
            color = 'black'
            title = f"Label: {true_label}"
        
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Hide extra subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(
    images: List[np.ndarray],
    labels: List[str],
    probabilities: List[np.ndarray],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize images with predicted labels and probability bars.
    
    Args:
        images: List of images as numpy arrays
        labels: List of predicted class names
        probabilities: List of probability arrays
        save_path: Path to save visualization
        
    Example:
        visualize_predictions(
            images=[img1, img2],
            labels=["Benign", "Suspicious"],
            probabilities=[probs1, probs2]
        )
    """
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 3))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_images):
        # Plot image
        axes[idx, 0].imshow(images[idx])
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title(f"Prediction: {labels[idx]}", fontweight='bold')
        
        # Plot probability bar
        axes[idx, 1].barh(CLASS_NAMES, probabilities[idx], color='skyblue')
        axes[idx, 1].set_xlim(0, 1)
        axes[idx, 1].set_xlabel('Probability', fontweight='bold')
        axes[idx, 1].set_title('Class Probabilities', fontweight='bold')
        axes[idx, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_sample_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    cols: int = 4
) -> None:
    """
    Create a grid of images.
    
    Args:
        images: List of images as numpy arrays
        titles: Optional list of titles for each image
        save_path: Path to save grid
        cols: Number of columns
        
    Example:
        create_sample_grid(
            images=[img1, img2, img3, img4],
            titles=["Sample 1", "Sample 2", "Sample 3", "Sample 4"],
            cols=2
        )
    """
    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, image in enumerate(images):
        axes[idx].imshow(image)
        axes[idx].axis('off')
        
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx], fontweight='bold')
    
    # Hide extra subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grid saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
        
    Example:
        info = get_device_info()
        print(f"Using: {info['device_name']}")
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': 'CPU'
    }
    
    if info['cuda_available']:
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
        info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1e9
        info['memory_reserved'] = torch.cuda.memory_reserved(0) / 1e9
        info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info


def print_device_info() -> None:
    """Print device information to console."""
    info = get_device_info()
    
    print("\n" + "=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)
    print(f"Device: {info['device_name']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Memory: {info['memory_total']:.2f} GB total")
        print(f"  Allocated: {info['memory_allocated']:.2f} GB")
        print(f"  Reserved: {info['memory_reserved']:.2f} GB")
    
    print("=" * 70 + "\n")


def count_files_by_class(data_dir: Union[str, Path]) -> dict:
    """
    Count number of files in each class folder.
    
    Args:
        data_dir: Root directory containing class folders
        
    Returns:
        Dictionary mapping class names to file counts
        
    Example:
        counts = count_files_by_class("data/train")
        print(f"Benign: {counts['Benign']} images")
    """
    data_path = Path(data_dir)
    counts = {}
    
    for class_name in CLASS_NAMES:
        class_dir = data_path / class_name
        if class_dir.exists():
            # Count image files
            count = len(list(class_dir.glob('*.jpg'))) + \
                    len(list(class_dir.glob('*.jpeg'))) + \
                    len(list(class_dir.glob('*.png')))
            counts[class_name] = count
        else:
            counts[class_name] = 0
    
    return counts


if __name__ == "__main__":
    # Test utilities
    print("Testing utils module...")
    
    # Test seed setting
    set_global_seed(42)
    print(f"Random value: {random.random()}")
    
    # Test device info
    print_device_info()
    
    # Test image operations
    try:
        # Create dummy image
        dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Test denormalize
        normalized = (dummy_image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        denormalized = denormalize_image(normalized)
        
        print(f"Denormalized image shape: {denormalized.shape}")
        print(f"Denormalized image dtype: {denormalized.dtype}")
        
        print("\nUtils module test passed!")
        
    except Exception as e:
        print(f"Error testing utils: {e}")
