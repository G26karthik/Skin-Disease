"""
Preprocess ISIC dataset for training.

This script converts ISIC metadata and images into the required
directory structure for training.

Dataset structure expected:
    ISIC_raw/
        images/
            ISIC_0000001.jpg
            ISIC_0000002.jpg
            ...
        metadata.csv (with columns: image_name, diagnosis, split)

Output structure:
    data/
        train/
            Benign/
            Suspicious/
            Urgent/
        val/
            ...
        test/
            ...

Usage:
    python scripts/preprocess_isic.py --input_dir ISIC_raw --output_dir data
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter


# Mapping from ISIC diagnosis to our classes
DIAGNOSIS_MAPPING = {
    'nevus': 'Benign',
    'nv': 'Benign',
    'melanoma': 'Urgent',
    'mel': 'Urgent',
    'basal cell carcinoma': 'Urgent',
    'bcc': 'Urgent',
    'squamous cell carcinoma': 'Urgent',
    'scc': 'Urgent',
    'actinic keratosis': 'Suspicious',
    'ak': 'Suspicious',
    'akiec': 'Suspicious',
    'benign keratosis': 'Benign',
    'bkl': 'Benign',
    'dermatofibroma': 'Benign',
    'df': 'Benign',
    'vascular lesion': 'Suspicious',
    'vasc': 'Suspicious',
}


def parse_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Parse ISIC metadata CSV.
    
    Args:
        metadata_path: Path to metadata CSV file
        
    Returns:
        DataFrame with image_name, diagnosis, and split columns
    """
    print(f"Reading metadata from {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Check required columns
    required_cols = ['image_name', 'diagnosis']
    for col in required_cols:
        if col not in df.columns:
            # Try common alternatives
            if 'image' in df.columns and col == 'image_name':
                df['image_name'] = df['image']
            elif 'dx' in df.columns and col == 'diagnosis':
                df['diagnosis'] = df['dx']
            elif 'image_id' in df.columns and col == 'image_name':
                df['image_name'] = df['image_id']
            else:
                raise ValueError(f"Required column '{col}' not found in metadata")
    
    # Map diagnoses to our classes
    df['class'] = df['diagnosis'].str.lower().map(DIAGNOSIS_MAPPING)
    
    # Remove unmapped diagnoses
    unmapped = df[df['class'].isna()]
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} images have unmapped diagnoses")
        print(f"Unmapped diagnoses: {unmapped['diagnosis'].unique()}")
        df = df[df['class'].notna()]
    
    print(f"Found {len(df)} images with valid diagnoses")
    print(f"Class distribution: {dict(Counter(df['class']))}")
    
    return df


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True
) -> pd.DataFrame:
    """
    Create train/val/test splits.
    
    Args:
        df: DataFrame with image metadata
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        stratify: Whether to stratify by class
        
    Returns:
        DataFrame with 'split' column added
    """
    from sklearn.model_selection import train_test_split
    
    # Check if split already exists
    if 'split' in df.columns:
        print("Using existing split information")
        return df
    
    print("Creating train/val/test splits...")
    
    if stratify:
        # Stratified split
        train_val, test = train_test_split(
            df, test_size=test_ratio, stratify=df['class'], random_state=42
        )
        
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, test_size=val_size, stratify=train_val['class'], random_state=42
        )
    else:
        # Random split
        train_val, test = train_test_split(
            df, test_size=test_ratio, random_state=42
        )
        
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, test_size=val_size, random_state=42
        )
    
    # Assign splits
    df.loc[train.index, 'split'] = 'train'
    df.loc[val.index, 'split'] = 'val'
    df.loc[test.index, 'split'] = 'test'
    
    print(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return df


def organize_images(
    df: pd.DataFrame,
    input_images_dir: Path,
    output_dir: Path
):
    """
    Copy images to organized directory structure.
    
    Args:
        df: DataFrame with image metadata and splits
        input_images_dir: Directory containing source images
        output_dir: Directory to create organized structure
    """
    print(f"\nOrganizing images from {input_images_dir} to {output_dir}")
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        for class_name in ['Benign', 'Suspicious', 'Urgent']:
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    copied = 0
    skipped = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying images"):
        image_name = row['image_name']
        class_name = row['class']
        split = row['split']
        
        # Find source image (handle various extensions)
        source_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = input_images_dir / f"{image_name}{ext}"
            if candidate.exists():
                source_path = candidate
                break
        
        if source_path is None:
            skipped += 1
            continue
        
        # Destination path
        dest_path = output_dir / split / class_name / f"{image_name}{source_path.suffix}"
        
        # Copy image
        shutil.copy2(source_path, dest_path)
        copied += 1
    
    print(f"\n✓ Copied {copied} images")
    if skipped > 0:
        print(f"⚠ Skipped {skipped} images (not found)")
    
    # Print summary
    print("\nDataset summary:")
    for split in ['train', 'val', 'test']:
        print(f"  {split}:")
        for class_name in ['Benign', 'Suspicious', 'Urgent']:
            class_dir = output_dir / split / class_name
            count = len(list(class_dir.glob('*')))
            print(f"    {class_name}: {count} images")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Preprocess ISIC dataset')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing ISIC raw data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for organized data'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='metadata.csv',
        help='Name of metadata CSV file'
    )
    parser.add_argument(
        '--images_subdir',
        type=str,
        default='images',
        help='Subdirectory containing images'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    metadata_path = input_path / args.metadata
    images_dir = input_path / args.images_subdir
    
    # Validate paths
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Parse metadata
    df = parse_metadata(metadata_path)
    
    # Create splits
    df = create_splits(df)
    
    # Organize images
    organize_images(df, images_dir, output_path)
    
    print(f"\n✓ Dataset preprocessing complete!")
    print(f"  Output directory: {output_path}")
    print(f"\nYou can now train the model with:")
    print(f"  python -m src.train --data_dir {output_path} --epochs 20")


if __name__ == "__main__":
    main()
