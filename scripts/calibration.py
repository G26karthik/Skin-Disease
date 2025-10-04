"""Calibration metrics and reliability diagram generation.

Computes Expected Calibration Error (ECE), Brier Score, and produces a reliability diagram.
Uses validation split predictions from current trained model.

Outputs:
  runs/calibration.json
  docs/images/reliability.png

Usage:
  python -m scripts.calibration --data_dir data/ham10000 --model_path models/model.pt --bins 15
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F

from src.config import DEVICE, CLASS_NAMES
from src.dataset import get_dataloaders
from src.model_builder import build_model

RUNS = Path('runs'); RUNS.mkdir(exist_ok=True)
IMG_DIR = Path('docs/images'); IMG_DIR.mkdir(parents=True, exist_ok=True)

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0,1,bins+1)
    ece = 0.0
    bin_data = []
    for i in range(bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i+1]) if i < bins-1 else (confidences >= bin_edges[i])
        if mask.sum() == 0:
            bin_data.append({'bin': i, 'conf': None, 'acc': None, 'count': 0})
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = (predictions[mask] == labels[mask]).mean()
        weight = mask.mean()
        ece += weight * abs(bin_acc - bin_conf)
        bin_data.append({'bin': i, 'conf': float(bin_conf), 'acc': float(bin_acc), 'count': int(mask.sum())})
    return float(ece), bin_data

def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return float(((probs - one_hot)**2).sum(axis=1).mean())

@torch.no_grad()
def collect_probs(model: torch.nn.Module, loader):
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.concatenate(all_labels)

def reliability_diagram(bin_data, save_path: Path):
    # Filter valid bins
    valid = [b for b in bin_data if b['count']>0]
    if not valid:
        return
    confs = [b['conf'] for b in valid]
    accs = [b['acc'] for b in valid]
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], '--', color='gray', label='Perfect')
    plt.plot(confs, accs, marker='o', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model_path', default='models/model.pt')
    ap.add_argument('--bins', type=int, default=15)
    args = ap.parse_args()

    # Data
    _, val_loader, _ = get_dataloaders(args.data_dir, batch_size=64)

    # Model
    model = build_model()
    state = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.to(DEVICE)

    probs, labels = collect_probs(model, val_loader)
    ece, bin_data = expected_calibration_error(probs, labels, bins=args.bins)
    brier = brier_score(probs, labels, num_classes=len(CLASS_NAMES))

    out = {
        'ece': ece,
        'brier': brier,
        'bins': bin_data
    }
    with open(RUNS / 'calibration.json','w') as f:
        json.dump(out, f, indent=2)
    reliability_diagram(bin_data, IMG_DIR / 'reliability.png')
    print(json.dumps({'ece': ece, 'brier': brier}, indent=2))

if __name__ == '__main__':
    main()
