"""Export representative Grad-CAM visualizations for each class.

Selects up to N images per class from validation split, runs model inference,
computes Grad-CAM overlays, and saves composite tiles to docs/gradcam_samples.

Usage:
    python scripts/export_gradcam_samples.py --data_dir data/ham10000 --per_class 1
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from src.config import DEVICE, CLASS_NAMES
from src.dataset import get_dataloaders
from src.model_builder import build_model
from src.gradcam import GradCAM, create_gradcam_overlay

OUT_DIR = Path('docs/gradcam_samples')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Basic tensor -> image normalization inverse
IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
IMAGENET_STD = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

def tensor_to_uint8(img: torch.Tensor):
    img = img.cpu()*IMAGENET_STD + IMAGENET_MEAN
    img = torch.clamp(img,0,1)
    return (img.permute(1,2,0).numpy()*255).astype(np.uint8)

@torch.no_grad()
def collect_samples(loader, num_per_class):
    buckets = {i: [] for i in range(len(CLASS_NAMES))}
    for images, labels in loader:
        for img, lbl in zip(images, labels):
            if len(buckets[lbl.item()]) < num_per_class:
                buckets[lbl.item()].append(img)
        if all(len(v)>=num_per_class for v in buckets.values()):
            break
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_path', default='models/model.pt')
    parser.add_argument('--per_class', type=int, default=1)
    args = parser.parse_args()

    # Load data
    _, val_loader, _ = get_dataloaders(args.data_dir, batch_size=32)

    # Load model
    model = build_model()
    state = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.to(DEVICE).eval()

    gradcam = GradCAM(model)

    samples = collect_samples(val_loader, args.per_class)

    for class_idx, imgs in samples.items():
        class_name = CLASS_NAMES[class_idx]
        for i, tensor_img in enumerate(imgs):
            inp = tensor_img.unsqueeze(0).to(DEVICE)
            heatmap = gradcam.generate_heatmap(inp, target_class=class_idx)
            np_img = tensor_to_uint8(tensor_img)
            overlay = create_gradcam_overlay(np_img, heatmap)
            composite = np.concatenate([np_img, overlay], axis=1)
            out_path = OUT_DIR / f"{class_name}_{i+1}.png"
            Image.fromarray(composite).save(out_path)
            print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
