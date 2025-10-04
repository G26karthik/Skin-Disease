"""
Preprocess HAM10000 dataset for training.

This script converts the HAM10000 Kaggle dataset into the required
directory structure for training with proper train/val/test splits.

HAM10000 Dataset Classes:
    - nv: Melanocytic nevi (benign)
    - mel: Melanoma (malignant)
    - bkl: Benign keratosis-like lesions
    - bcc: Basal cell carcinoma (malignant)
    - akiec: Actinic keratoses and intraepithelial carcinoma (pre-cancerous)
    - vasc: Vascular lesions (benign)
    - df: Dermatofibroma (benign)

Dataset: 10,015 images across 7 classes
Source: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Usage:
    python scripts/preprocess_ham10000.py
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
import os


# HAM10000 7-class system (keeping original diagnostic labels)
HAM10000_CLASSES = {
    'nv': 'Melanocytic_nevi',           # Benign moles
    'mel': 'Melanoma',                  # Malignant melanoma
    'bkl': 'Benign_keratosis',          # Benign keratosis lesions
    'bcc': 'Basal_cell_carcinoma',      # Malignant basal cell carcinoma
    'akiec': 'Actinic_keratoses',       # Pre-cancerous/in-situ carcinoma
    'vasc': 'Vascular_lesions',         # Benign vascular lesions
    'df': 'Dermatofibroma'              # Benign dermatofibroma
}


def parse_metadata(metadata_path, img_dir1, img_dir2):
    """Parse HAM10000 metadata and locate images."""
    print(f"\nReading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    print(f"Total images in metadata: {len(df)}")
    print(f"\nClass distribution:")
    print(df['dx'].value_counts())
    
    # Map to full class names
    df['class_name'] = df['dx'].map(HAM10000_CLASSES)
    
    # Find image paths (images split across two folders)
    image_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Locating images"):
        image_id = row['image_id']
        img_path1 = img_dir1 / f"{image_id}.jpg"
        img_path2 = img_dir2 / f"{image_id}.jpg"
        
        if img_path1.exists():
            image_paths.append(str(img_path1))
        elif img_path2.exists():
            image_paths.append(str(img_path2))
        else:
            image_paths.append(None)
            print(f"Warning: Image not found: {image_id}")
    
    df['image_path'] = image_paths
    
    # Remove missing images
    df = df.dropna(subset=['image_path'])
    print(f"\nTotal images found: {len(df)}")
    
    return df


def stratified_split(df, val_size=0.15, test_size=0.15, random_state=42):
    """Create stratified train/val/test split ensuring same lesion doesn't appear in multiple sets."""
    print("\nCreating stratified splits...")
    
    # Group by lesion_id to ensure same lesion doesn't leak across splits
    lesion_groups = df.groupby('lesion_id')
    
    # Get one representative per lesion for splitting
    lesion_representatives = lesion_groups.first().reset_index()
    
    # First split: train+val vs test
    train_val_lesions, test_lesions = train_test_split(
        lesion_representatives['lesion_id'],
        test_size=test_size,
        stratify=lesion_representatives['dx'],
        random_state=random_state
    )
    
    # Second split: train vs val
    train_val_df = lesion_representatives[lesion_representatives['lesion_id'].isin(train_val_lesions)]
    val_ratio = val_size / (1 - test_size)
    
    train_lesions, val_lesions = train_test_split(
        train_val_df['lesion_id'],
        test_size=val_ratio,
        stratify=train_val_df['dx'],
        random_state=random_state
    )
    
    # Assign splits to all images based on their lesion_id
    df['split'] = 'train'
    df.loc[df['lesion_id'].isin(val_lesions), 'split'] = 'val'
    df.loc[df['lesion_id'].isin(test_lesions), 'split'] = 'test'
    
    print(f"\nSplit distribution:")
    print(df.groupby(['split', 'dx']).size().unstack(fill_value=0))
    
    return df


def organize_images(df, output_dir):
    """Copy images to train/val/test directory structure."""
    output_dir = Path(output_dir)
    
    splits = ['train', 'val', 'test']
    classes = list(HAM10000_CLASSES.values())
    
    # Create directory structure
    print("\nCreating directory structure...")
    for split in splits:
        for class_name in classes:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Copy images
    print("\nOrganizing images...")
    for split in splits:
        split_df = df[df['split'] == split]
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split}"):
            src_path = Path(row['image_path'])
            class_name = row['class_name']
            image_id = row['image_id']
            
            dst_path = output_dir / split / class_name / f"{image_id}.jpg"
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
    
    # Print summary
    print("\n" + "="*70)
    print("DATASET ORGANIZATION COMPLETE")
    print("="*70)
    
    for split in splits:
        print(f"\n{split.upper()} SET:")
        split_df = df[df['split'] == split]
        class_counts = split_df['dx'].value_counts()
        for class_code, count in class_counts.items():
            class_name = HAM10000_CLASSES[class_code]
            print(f"  {class_name}: {count} images")
        print(f"  Total: {len(split_df)} images")
    
    print(f"\nDataset location: {output_dir.absolute()}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Preprocess HAM10000 dataset')
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='Directory containing raw HAM10000 data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/ham10000',
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.15,
        help='Validation set size (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.15,
        help='Test set size (default: 0.15 = 15%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    raw_dir = Path(args.raw_dir)
    metadata_path = raw_dir / 'HAM10000_metadata.csv'
    img_dir1 = raw_dir / 'HAM10000_images_part_1'
    img_dir2 = raw_dir / 'HAM10000_images_part_2'
    output_dir = Path(args.output_dir)
    
    # Validate input
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not img_dir1.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir1}")
    if not img_dir2.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir2}")
    
    print("="*70)
    print("HAM10000 DATASET PREPROCESSING")
    print("="*70)
    print(f"Source: {raw_dir.absolute()}")
    print(f"Destination: {output_dir.absolute()}")
    print(f"Splits: {(1-args.val_size-args.test_size)*100:.0f}% train, "
          f"{args.val_size*100:.0f}% val, {args.test_size*100:.0f}% test")
    print("="*70)
    
    # Parse metadata and locate images
    df = parse_metadata(metadata_path, img_dir1, img_dir2)
    
    # Create stratified splits
    df = stratified_split(df, args.val_size, args.test_size, args.seed)
    
    # Organize images into directory structure
    organize_images(df, output_dir)
    
    # Save processed metadata
    metadata_output = output_dir / 'processed_metadata.csv'
    df[['image_id', 'lesion_id', 'dx', 'class_name', 'split', 'age', 'sex', 'localization']].to_csv(
        metadata_output, index=False
    )
    print(f"\nProcessed metadata saved to: {metadata_output}")
    
    print("\n✓ Dataset ready for training!")
    print(f"\nTo start training:")
    print(f"  python -m src.train --data_dir {output_dir} --epochs 30")


if __name__ == '__main__':
    main()
