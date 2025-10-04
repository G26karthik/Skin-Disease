"""Evaluate trained model on validation or test set and produce metrics artifacts.

Outputs:
  runs/eval_metrics.json  (all aggregate + per-class metrics)
  models/confusion_matrix.npy (raw counts)
  docs/images/per_class_metrics.png (bar chart)
  docs/images/confusion_matrix.png (normalized heatmap override)
  runs/throughput.json (latency + throughput stats)

Usage:
    python scripts/evaluate_model.py --data_dir data/ham10000 --split val

"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import DEVICE, CLASS_NAMES
from src.dataset import get_dataloaders
from src.model_builder import build_model
from src.metrics import calculate_metrics, plot_confusion_matrix, plot_per_class_metrics

DOC_IMG = Path('docs/images')
DOC_IMG.mkdir(parents=True, exist_ok=True)
RUNS = Path('runs'); RUNS.mkdir(exist_ok=True)
MODELS = Path('models')

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader):
    model.eval()
    all_preds = []
    all_labels = []
    start = time.time()
    total_images = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        total_images += images.size(0)
    duration = time.time() - start
    ips = total_images / duration
    return np.array(all_labels), np.array(all_preds), duration, ips


def benchmark_latency(model: torch.nn.Module, sample: torch.Tensor, runs: int = 50):
    torch.cuda.synchronize() if DEVICE.type=='cuda' else None
    timings = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.time()
            _ = model(sample)
            torch.cuda.synchronize() if DEVICE.type=='cuda' else None
            timings.append(time.time() - t0)
    arr = np.array(timings[5:])  # warmup drop first 5
    return {
        'mean_latency_s': float(arr.mean()),
        'p50_s': float(np.percentile(arr,50)),
        'p95_s': float(np.percentile(arr,95)),
        'p99_s': float(np.percentile(arr,99)),
        'throughput_fps': float(1.0/arr.mean()),
        'runs': len(arr)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_path', default='models/model.pt')
    args = parser.parse_args()

    # Load loaders (reuse existing function: returns train,val,test)
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    loader = val_loader if args.split=='val' else test_loader

    # Load model
    model = build_model()
    state = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.to(DEVICE)
    model.eval()

    y_true, y_pred, duration, ips = evaluate(model, loader)
    metrics = calculate_metrics(y_true, y_pred)

    # Save raw confusion matrix counts
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    np.save(MODELS / 'confusion_matrix.npy', cm)

    # Plots
    plot_confusion_matrix(y_true, y_pred, save_path=str(DOC_IMG / 'confusion_matrix.png'))
    plot_per_class_metrics(metrics, save_path=str(DOC_IMG / 'per_class_metrics.png'))

    metrics_out = {
        'split': args.split,
        'num_samples': int(len(y_true)),
        'overall': {
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall']
        },
        'per_class': metrics['per_class'],
        'evaluation_time_s': duration,
        'images_per_second': ips
    }

    with open(RUNS / 'eval_metrics.json','w') as f:
        json.dump(metrics_out, f, indent=2)

    # Latency benchmark (batch=1)
    sample = next(iter(loader))[0][:1].to(DEVICE)
    latency_stats = benchmark_latency(model, sample)
    with open(RUNS / 'throughput.json','w') as f:
        json.dump(latency_stats, f, indent=2)

    print('Evaluation complete:')
    print(json.dumps({**metrics_out['overall'], 'latency_mean_s': latency_stats['mean_latency_s']}, indent=2))

if __name__ == '__main__':
    main()
