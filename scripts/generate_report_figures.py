"""Generate report-quality figures for README and documentation.

This script creates:
1. Class distribution bar chart
2. Sample architecture diagram (ASCII rendered onto an image)
3. Placeholder training curves (if real log not parsed)
4. Placeholder confusion matrix (uniform or skewed example)
5. Workflow diagram
6. Roadmap timeline graphic
7. Grad-CAM montage placeholder (until real outputs saved)

All images saved under docs/images/.
Real data hooks:
- If runs/last_run.json exists with metrics history arrays, they will be plotted.
- If models/confusion_matrix.npy exists, it will be used instead of placeholder.
- If docs/gradcam_samples/ contains images, they will be assembled into a grid.

Usage:
    python scripts/generate_report_figures.py

Dependencies: matplotlib, seaborn, numpy, pillow, pandas (optional), json, textwrap
"""
from __future__ import annotations
import json
import os
import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / 'docs' / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    'nv','mel','bkl','bcc','akiec','vasc','df'
]
CLASS_COUNTS = [6705,1113,1099,514,327,142,115]

PALETTE = sns.color_palette('viridis', len(CLASS_NAMES))


def save_bar_distribution():
    plt.figure(figsize=(8,4))
    sns.barplot(x=CLASS_NAMES, y=CLASS_COUNTS, palette=PALETTE)
    plt.title('HAM10000 Class Distribution')
    plt.ylabel('Image Count')
    plt.xlabel('Class')
    for i,v in enumerate(CLASS_COUNTS):
        plt.text(i, v+50, str(v), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    (IMG_DIR / 'class_distribution.png').unlink(missing_ok=True)
    plt.savefig(IMG_DIR / 'class_distribution.png', dpi=180)
    plt.close()


def _wrap(txt, width=50):
    return '\n'.join(textwrap.wrap(txt, width=width))


def architecture_diagram():
    """Render a simple architecture block diagram as an image."""
    w,h = 900, 500
    img = Image.new('RGB', (w,h), 'white')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 16)
        font_b = ImageFont.truetype('arial.ttf', 20)
    except IOError:
        font = ImageFont.load_default(); font_b = font

    title = 'EfficientNet_B0 Dermatology Pipeline'
    d.text((w//2, 20), title, anchor='mm', font=font_b, fill='black')

    # Blocks definitions
    blocks = [
        ('Input 224x224 RGB', (50,120,220,190)),
        ('Augmentation\nFlip/Rotate/Color', (250,120,420,190)),
        ('EfficientNet_B0\nPretrained Backbone', (450,80,720,230)),
        ('GlobalAvgPool', (760,110,880,160)),
        ('Dropout 0.3', (760,170,880,220)),
        ('FC 1280→7 + Softmax', (740,250,900,330)),
    ]

    for label, rect in blocks:
        d.rectangle(rect, outline='black', width=2)
        cx = (rect[0]+rect[2])//2
        cy = (rect[1]+rect[3])//2
        d.text((cx, cy), label, anchor='mm', font=font, fill='black', align='center')

    # Arrows
    def arrow(x1,y1,x2,y2):
        d.line((x1,y1,x2,y2), fill='black', width=3)
        # arrow head
        d.polygon([(x2,y2),(x2-8,y2-5),(x2-8,y2+5)], fill='black')

    arrow(220,155,250,155)
    arrow(420,155,450,155)
    arrow(720,155,760,135)
    arrow(720,155,760,195)
    arrow(720,155,740,290)

    d.text((w-10,h-10), 'Generated', anchor='rd', font=font, fill='gray')
    img.save(IMG_DIR / 'architecture.png')


def _smooth(arr, window=3):
    if arr is None: return None
    if len(arr) < window or window <=1:
        return arr
    out = []
    for i in range(len(arr)):
        seg = arr[max(0,i-window+1):i+1]
        out.append(sum(seg)/len(seg))
    return out

def training_curves():
    # Try to parse a metrics log if exists
    acc = None; val_acc=None
    history_path = BASE / 'runs' / 'history.json'
    if history_path.exists():
        try:
            with open(history_path,'r') as f: hist = json.load(f)
            acc = hist.get('train_accuracy'); val_acc = hist.get('val_accuracy')
        except Exception:
            pass
    if acc is None:
        epochs = np.arange(1,11)
        acc = 0.55 + 0.03*np.log1p(epochs)
        val_acc = acc - 0.02 + 0.01*np.sin(epochs)
    # Apply smoothing if enough epochs
    if len(acc) >= 5:
        acc_s = _smooth(acc, window=3)
        val_acc_s = _smooth(val_acc, window=3)
    else:
        acc_s, val_acc_s = acc, val_acc
    plt.figure(figsize=(7,4))
    plt.plot(acc_s, label='Train Acc')
    plt.plot(val_acc_s, label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'training_curves.png', dpi=180)
    plt.close()


def confusion_matrix_fig():
    cm_path = BASE / 'models' / 'confusion_matrix.npy'
    if cm_path.exists():
        cm = np.load(cm_path)
    else:
        # Placeholder simulated confusion matrix
        rng = np.random.default_rng(42)
        base = rng.uniform(0.0, 0.15, size=(len(CLASS_NAMES), len(CLASS_NAMES)))
        np.fill_diagonal(base, 0.55 + rng.uniform(0.05,0.15,size=len(CLASS_NAMES)))
        cm = base / base.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='magma', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'confusion_matrix.png', dpi=180)
    plt.close()


def workflow_diagram():
    w,h = 900, 260
    img = Image.new('RGB', (w,h), 'white')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except IOError:
        font = ImageFont.load_default()
    steps = [
        'Image Capture', 'Preprocess', 'Model Inference', 'Grad-CAM', 'Threshold Logic', 'Clinician Review', 'Decision Record'
    ]
    x = 50
    for s in steps:
        box = (x,80,x+110,160)
        d.rectangle(box, outline='black', width=2)
        d.text(( (box[0]+box[2])//2, (box[1]+box[3])//2 ), _wrap(s,12), anchor='mm', font=font, fill='black', align='center')
        x += 120
    # arrows
    x = 160
    for _ in range(len(steps)-1):
        d.polygon([(x,120),(x+18,120-6),(x+18,120+6)], fill='black')
        x += 120
    img.save(IMG_DIR / 'workflow.png')


def roadmap_timeline():
    phases = [
        ('Phase 1','0-6M','Baseline\nValidation'),
        ('Phase 2','6-12M','Regulatory\nPrep'),
        ('Phase 3','12-24M','Multimodal\nEnsembles'),
        ('Phase 4','24M+','Precision\nPlatform')
    ]
    w,h = 900, 250
    img = Image.new('RGB',(w,h),'white')
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 16)
        font_b = ImageFont.truetype('arial.ttf', 18)
    except IOError:
        font = ImageFont.load_default(); font_b=font
    d.text((w//2,25),'Strategic Roadmap',anchor='mm',font=font_b,fill='black')
    y = 140
    x_positions = np.linspace(80,w-80,len(phases))
    d.line((80,y,w-80,y), fill='black', width=3)
    for (label,period,desc),x in zip(phases,x_positions):
        d.ellipse((x-18,y-18,x+18,y+18),outline='black',width=3)
        d.text((x,y+35), label+'\n'+period, anchor='mm', font=font, fill='black')
        d.text((x,y-60), desc, anchor='mm', font=font, fill='black')
    img.save(IMG_DIR / 'roadmap.png')


def gradcam_placeholder():
    # If real gradcam samples folder exists use them
    sample_dir = BASE / 'docs' / 'gradcam_samples'
    out_path = IMG_DIR / 'gradcam_montage.png'
    images: List[Image.Image] = []
    if sample_dir.exists():
        for p in sorted(sample_dir.glob('*.png'))[:6]:
            try:
                images.append(Image.open(p).resize((224,224)))
            except Exception:
                continue
    if not images:
        # create colored placeholders with varying hues
        base_colors = [
            (220, 235, 250),
            (230, 220, 250),
            (250, 230, 220),
            (235, 250, 225),
            (250, 245, 210),
            (225, 240, 245)
        ]
        for i in range(6):
            img = Image.new('RGB',(224,224), base_colors[i])
            d = ImageDraw.Draw(img)
            d.text((112,112), f'Grad-CAM\nSample {i+1}', anchor='mm', fill='black', align='center')
            images.append(img)
    # assemble grid 3x2
    grid_w, grid_h = 224*3, 224*2
    canvas = Image.new('RGB',(grid_w,grid_h),'white')
    for idx,im in enumerate(images):
        r = idx//3; c = idx%3
        canvas.paste(im,(c*224,r*224))
    canvas.save(out_path)

def throughput_plot():
    tp_json = BASE / 'runs' / 'throughput.json'
    if not tp_json.exists():
        return
    with open(tp_json,'r') as f:
        data = json.load(f)
    labels = ['p50','p95','p99','mean']
    vals = [data.get('p50_s'), data.get('p95_s'), data.get('p99_s'), data.get('mean_latency_s')]
    plt.figure(figsize=(5,4))
    sns.barplot(x=labels, y=vals, palette='crest')
    plt.ylabel('Seconds')
    plt.title('Inference Latency (Batch=1)')
    for i,v in enumerate(vals):
        plt.text(i, v + max(vals)*0.02, f"{v:.3f}s", ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'latency.png', dpi=160)
    plt.close()

def per_class_metrics_plot():
    eval_json = BASE / 'runs' / 'test_eval_metrics.json'
    if not eval_json.exists():
        return
    with open(eval_json,'r') as f:
        data = json.load(f)
    per = data.get('per_class', {})
    if not per:
        return
    classes = list(per.keys())
    precisions = [per[c]['precision'] for c in classes]
    recalls = [per[c]['recall'] for c in classes]
    f1s = [per[c]['f1'] for c in classes]
    x = np.arange(len(classes))
    w = 0.25
    plt.figure(figsize=(10,4))
    plt.bar(x-w, precisions, w, label='Precision')
    plt.bar(x, recalls, w, label='Recall')
    plt.bar(x+w, f1s, w, label='F1')
    plt.xticks(x, classes, rotation=20)
    plt.ylim(0,1.05)
    plt.ylabel('Score')
    plt.title('Per-Class Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'per_class_metrics.png', dpi=180)
    plt.close()


def main():
    save_bar_distribution()
    architecture_diagram()
    training_curves()
    confusion_matrix_fig()
    workflow_diagram()
    roadmap_timeline()
    gradcam_placeholder()
    per_class_metrics_plot()
    throughput_plot()
    # Reliability diagram might already be produced by calibration script; just acknowledge if present.
    rel_path = IMG_DIR / 'reliability.png'
    if rel_path.exists():
        print('Found reliability diagram.')
    print('Figures generated in', IMG_DIR)

if __name__ == '__main__':
    main()
