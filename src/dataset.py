"""
Dataset module for Skin Lesion Classification.

This module provides PyTorch Dataset classes and data loading utilities
with augmentation for training, validation, and testing.

Example usage:
    from src.dataset import SkinLesionDataset, get_dataloaders
    
    train_loader, val_loader = get_dataloaders(
        data_dir="data",
        batch_size=32
    )
    
    for images, labels in train_loader:
        # Training loop
        pass
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, 
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    AUGMENTATION_CONFIG, CLASS_NAMES, SEED
)


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for skin lesion images.
    
    Expects directory structure:
        root/
            train/
                Benign/
                    img1.jpg
                    img2.jpg
                Suspicious/
                    img3.jpg
                Urgent/
                    img4.jpg
            val/
                ...
            test/
                ...
    
    Args:
        root: Root directory containing class subdirectories
        transform: Albumentations or torchvision transform
        split: One of 'train', 'val', 'test'
        class_names: List of class names (folder names)
        
    Example:
        dataset = SkinLesionDataset(
            root="data/train",
            transform=get_train_transforms(),
            split="train"
        )
        image, label = dataset[0]
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        class_names: List[str] = CLASS_NAMES
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.class_names = class_names
        
        # Build class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root}. Please check the directory structure.")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image paths and corresponding labels."""
        samples = []
        
        for class_name in self.class_names:
            class_dir = self.root / class_name
            
            if not class_dir.exists():
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue
            
            # Get all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    label = self.class_to_idx[class_name]
                    samples.append((img_path, label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label at index.
        
        Returns:
            image: Tensor of shape (3, H, W)
            label: Integer class label
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                # torchvision transform
                image = Image.fromarray(image)
                image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {name: 0 for name in self.class_names}
        
        for _, label in self.samples:
            class_name = self.class_names[label]
            distribution[class_name] += 1
        
        return distribution


def get_train_transforms() -> A.Compose:
    """
    Get training augmentation transforms using Albumentations.
    
    Includes geometric and color augmentations for better generalization.
    
    Returns:
        Albumentations Compose object
        
    Example:
        transform = get_train_transforms()
        augmented = transform(image=image)
        image_tensor = augmented["image"]
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=AUGMENTATION_CONFIG["rotation_limit"], p=0.5),
        A.HorizontalFlip(p=AUGMENTATION_CONFIG["flip_prob"]),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=AUGMENTATION_CONFIG["brightness_limit"],
            contrast_limit=AUGMENTATION_CONFIG["contrast_limit"],
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=AUGMENTATION_CONFIG["hue_shift_limit"],
            sat_shift_limit=AUGMENTATION_CONFIG["saturation_shift_limit"],
            val_shift_limit=20,
            p=0.5
        ),
        A.RandomResizedCrop(
            IMAGE_SIZE, IMAGE_SIZE,
            scale=AUGMENTATION_CONFIG["crop_scale"],
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=8, max_height=16, max_width=16,
            fill_value=0, p=0.3
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Returns:
        Albumentations Compose object
        
    Example:
        transform = get_val_transforms()
        transformed = transform(image=image)
        image_tensor = transformed["image"]
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def create_split_from_single_folder(
    data_dir: str,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = SEED
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train/val/test splits from a single folder containing all data.
    
    Args:
        data_dir: Directory containing class subdirectories
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        test_ratio: Proportion for testing (default 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset, test_dataset
        
    Example:
        train_ds, val_ds, test_ds = create_split_from_single_folder("data/all")
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Create full dataset without transforms first
    full_dataset = SkinLesionDataset(
        root=data_dir,
        transform=None,
        split="full"
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    test_dataset.dataset.transform = get_val_transforms()
    
    print(f"Dataset split - Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    data_dir: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    single_folder: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Get train, validation, and optional test dataloaders.
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        single_folder: If True, expects single folder and creates splits
        
    Returns:
        train_loader, val_loader, test_loader (test_loader may be None)
        
    Example:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir="data",
            batch_size=32
        )
    """
    data_path = Path(data_dir)
    
    if single_folder or not (data_path / "train").exists():
        # Create splits from single folder
        print("Creating train/val/test splits from single folder...")
        train_dataset, val_dataset, test_dataset = create_split_from_single_folder(data_dir)
        
    else:
        # Load pre-split datasets
        print("Loading pre-split datasets...")
        train_dataset = SkinLesionDataset(
            root=data_path / "train",
            transform=get_train_transforms(),
            split="train"
        )
        
        val_dataset = SkinLesionDataset(
            root=data_path / "val",
            transform=get_val_transforms(),
            split="val"
        )
        
        test_path = data_path / "test"
        test_dataset = None
        if test_path.exists():
            test_dataset = SkinLesionDataset(
                root=test_path,
                transform=get_val_transforms(),
                split="test"
            )
    
    # Print class distributions
    print(f"\nTrain distribution: {train_dataset.dataset.get_class_distribution() if hasattr(train_dataset, 'dataset') else train_dataset.get_class_distribution()}")
    print(f"Val distribution: {val_dataset.dataset.get_class_distribution() if hasattr(val_dataset, 'dataset') else val_dataset.get_class_distribution()}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=num_workers > 0
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=num_workers > 0
        )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset module...")
    
    # Check if sample data exists
    from src.config import DATA_DIR
    sample_dir = DATA_DIR / "sample"
    
    if not sample_dir.exists():
        print(f"Sample data not found at {sample_dir}")
        print("Run scripts/create_dummy_data.py to generate sample data for testing")
    else:
        try:
            train_loader, val_loader, _ = get_dataloaders(
                data_dir=str(sample_dir),
                batch_size=4,
                single_folder=True
            )
            
            # Test loading a batch
            images, labels = next(iter(train_loader))
            print(f"\nBatch shape: {images.shape}")
            print(f"Labels: {labels}")
            print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
            print("\nDataset module test passed!")
            
        except Exception as e:
            print(f"Error testing dataset: {e}")
