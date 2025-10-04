"""
Create dummy synthetic data for development and testing.

This script generates synthetic skin lesion images for quick testing
without needing to download the full ISIC dataset.

Usage:
    python scripts/create_dummy_data.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random
from pathlib import Path
import argparse

from src.config import DATA_DIR, CLASS_NAMES, set_seed


def create_synthetic_lesion(
    size: tuple = (600, 600),
    lesion_type: str = "Benign"
) -> Image.Image:
    """
    Create a synthetic skin lesion image.
    
    Args:
        size: Image size (width, height)
        lesion_type: Type of lesion (Benign, Suspicious, Urgent)
        
    Returns:
        PIL Image
    """
    # Create base skin tone
    skin_tones = [
        (255, 220, 177),  # Light
        (241, 194, 125),  # Medium-light
        (224, 172, 105),  # Medium
        (198, 134, 66),   # Medium-dark
        (141, 85, 36),    # Dark
    ]
    
    skin_color = random.choice(skin_tones)
    img = Image.new('RGB', size, skin_color)
    draw = ImageDraw.Draw(img)
    
    # Add skin texture noise
    pixels = np.array(img)
    noise = np.random.randint(-15, 15, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    
    # Lesion characteristics based on type
    if lesion_type == "Benign":
        # Symmetric, uniform color, regular borders
        center = (size[0] // 2, size[1] // 2)
        radius = random.randint(40, 80)
        color = (139, 69, 19)  # Brown
        
        # Draw circular lesion
        draw.ellipse(
            [center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius],
            fill=color
        )
        
    elif lesion_type == "Suspicious":
        # Slightly asymmetric, varied color, somewhat irregular borders
        center = (size[0] // 2 + random.randint(-20, 20),
                  size[1] // 2 + random.randint(-20, 20))
        radius_x = random.randint(50, 90)
        radius_y = random.randint(40, 80)
        color1 = (139, 69, 19)
        color2 = (85, 40, 15)
        
        # Draw elliptical lesion
        draw.ellipse(
            [center[0] - radius_x, center[1] - radius_y,
             center[0] + radius_x, center[1] + radius_y],
            fill=color1
        )
        
        # Add darker region
        draw.ellipse(
            [center[0] - radius_x//2, center[1] - radius_y//2,
             center[0] + radius_x//2, center[1] + radius_y//2],
            fill=color2
        )
        
    else:  # Urgent
        # Asymmetric, multiple colors, irregular borders, larger
        center = (size[0] // 2 + random.randint(-30, 30),
                  size[1] // 2 + random.randint(-30, 30))
        
        # Draw irregular shape with multiple colors
        colors = [(85, 40, 15), (50, 20, 10), (139, 69, 19), (180, 100, 50)]
        
        for i in range(4):
            offset_x = random.randint(-40, 40)
            offset_y = random.randint(-40, 40)
            radius_x = random.randint(60, 100)
            radius_y = random.randint(60, 100)
            
            draw.ellipse(
                [center[0] + offset_x - radius_x,
                 center[1] + offset_y - radius_y,
                 center[0] + offset_x + radius_x,
                 center[1] + offset_y + radius_y],
                fill=random.choice(colors)
            )
    
    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img


def create_dummy_dataset(
    output_dir: Path,
    num_per_class: int = 20,
    split: bool = True
):
    """
    Create a dummy dataset with synthetic images.
    
    Args:
        output_dir: Directory to save images
        num_per_class: Number of images per class
        split: Whether to create train/val/test splits
    """
    print(f"Creating dummy dataset in {output_dir}")
    print(f"Generating {num_per_class} images per class...")
    
    if split:
        splits = {
            'train': int(num_per_class * 0.7),
            'val': int(num_per_class * 0.2),
            'test': num_per_class - int(num_per_class * 0.7) - int(num_per_class * 0.2)
        }
    else:
        splits = {'all': num_per_class}
    
    for split_name, split_count in splits.items():
        split_dir = output_dir / split_name
        
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Generating {split_count} {class_name} images for {split_name}...")
            
            for i in range(split_count):
                # Create synthetic image
                img = create_synthetic_lesion(
                    size=(600, 600),
                    lesion_type=class_name
                )
                
                # Save image
                img_path = class_dir / f"{class_name.lower()}_{split_name}_{i:03d}.jpg"
                img.save(img_path, quality=85)
    
    print(f"\n✓ Dummy dataset created successfully!")
    print(f"  Location: {output_dir}")
    print(f"  Total images: {sum(splits.values()) * len(CLASS_NAMES)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create dummy dataset')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(DATA_DIR / 'dummy'),
        help='Output directory for dummy data'
    )
    parser.add_argument(
        '--num_per_class',
        type=int,
        default=20,
        help='Number of images per class'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Create train/val/test splits'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create dataset
    output_path = Path(args.output_dir)
    create_dummy_dataset(
        output_dir=output_path,
        num_per_class=args.num_per_class,
        split=args.split
    )
    
    print(f"\nYou can now test the project with:")
    print(f"  python -m src.train --data_dir {output_path} --epochs 5")


if __name__ == "__main__":
    main()
