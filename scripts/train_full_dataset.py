"""
Train the model on the entire dataset (train + val combined).

This script combines the training and validation sets for maximum data utilization,
then evaluates on the test set.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_SCHEDULER, LR_MIN, USE_AMP, MODEL_PATH, RUNS_DIR,
    set_seed, print_config
)
from src.dataset import SkinLesionDataset, get_train_transforms, get_val_transforms
from src.model_builder import build_model, save_model, count_parameters
from src.metrics import calculate_metrics
from sklearn.metrics import confusion_matrix


def get_combined_train_loader(data_dir: str, batch_size: int = BATCH_SIZE):
    """Get dataloader combining train and val sets."""
    data_path = Path(data_dir)

    # Load train dataset
    train_dataset = SkinLesionDataset(
        root=data_path / "train",
        transform=get_train_transforms(),
        split="train"
    )

    # Load val dataset
    val_dataset = SkinLesionDataset(
        root=data_path / "val",
        transform=get_train_transforms(),  # Use train transforms for combined training
        split="val"
    )

    # Combine datasets
    combined_dataset = ConcatDataset([train_dataset, val_dataset])

    print(f"Combined training dataset size: {len(combined_dataset)}")
    print(f"Train distribution: {train_dataset.get_class_distribution()}")
    print(f"Val distribution: {val_dataset.get_class_distribution()}")

    # Create dataloader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    return train_loader


def get_test_loader(data_dir: str, batch_size: int = BATCH_SIZE):
    """Get test dataloader."""
    data_path = Path(data_dir)

    test_dataset = SkinLesionDataset(
        root=data_path / "test",
        transform=get_val_transforms(),
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    print(f"Test distribution: {test_dataset.get_class_distribution()}")

    return test_loader


class FullDatasetTrainer:
    """Trainer for training on full dataset."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = DEVICE,
        use_amp: bool = USE_AMP
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp

        self.scaler = GradScaler() if use_amp else None

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'learning_rates': []
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                acc = 100.0 * correct / total
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.2f}%'
                })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def train(self, epochs: int, scheduler=None) -> Dict:
        """Train for multiple epochs."""
        print(f"\nStarting training on full dataset for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Batch Size: {self.train_loader.batch_size}")
        print(f"Train Batches: {len(self.train_loader)}")

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['learning_rates'].append(current_lr)

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()

            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{epochs} - {epoch_time:.1f}s")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | LR: {current_lr:.6f}")

            # Save model every 5 epochs
            if (epoch + 1) % 5 == 0:
                save_model(
                    model=self.model,
                    save_path=str(MODEL_PATH.parent / f"model_epoch_{epoch+1}.pt"),
                    optimizer=self.optimizer,
                    epoch=epoch,
                    scaler=self.scaler
                )

        # Save final model
        save_model(
            model=self.model,
            save_path=str(MODEL_PATH),
            optimizer=self.optimizer,
            epoch=epochs-1,
            scaler=self.scaler
        )

        print(f"\nTraining completed!")
        return self.history


def plot_training_curves(history: Dict, save_path: str = "runs/training_curves_full.png"):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Full Dataset Training History', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'g-', linewidth=2)
    axes[1].set_title('Training Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(epochs, history['learning_rates'], 'orange', linewidth=2)
    axes[2].set_title('Learning Rate', fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def evaluate_on_test(model, test_loader):
    """Evaluate the trained model on test set."""
    model.eval()
    all_preds = []
    all_labels = []

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculate metrics
    metrics = calculate_metrics(
        y_true=np.array(all_labels),
        y_pred=np.array(all_preds)
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    np.save(Path('models') / 'confusion_matrix_full.npy', cm)

    print("\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")

    return metrics


def main():
    """Main training function."""
    # Set seed
    set_seed()
    print_config()

    data_dir = "data/ham10000"
    epochs = 30  # Full training on entire dataset
    batch_size = 32

    # Load combined train loader (train + val)
    train_loader = get_combined_train_loader(data_dir, batch_size)

    # Load test loader
    test_loader = get_test_loader(data_dir, batch_size)

    # Build model
    print("\nBuilding model...")
    model = build_model()
    print(f"Total parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=LR_MIN
    )

    # Create trainer
    trainer = FullDatasetTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        use_amp=USE_AMP
    )

    # Train
    history = trainer.train(epochs=epochs, scheduler=scheduler)

    # Plot curves
    plot_training_curves(history)

    # Save history
    history_path = RUNS_DIR / 'history_full.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    metrics = evaluate_on_test(model, test_loader)

    # Save results
    results = {
        'epochs': epochs,
        'batch_size': batch_size,
        'final_train_acc': history['train_acc'][-1],
        'test_accuracy': metrics['accuracy'],
        'test_macro_f1': metrics['macro_f1'],
        'test_macro_precision': metrics['macro_precision'],
        'test_macro_recall': metrics['macro_recall'],
    }

    results_path = RUNS_DIR / "full_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nFull training completed! Results saved to {results_path}")


if __name__ == "__main__":
    main()