"""
Training module for Skin Lesion Classifier.

This module provides training and validation loops with GPU optimization,
mixed precision training, checkpointing, and automatic batch size reduction
for OOM errors.

Example usage:
    python -m src.train --data_dir data --epochs 20 --batch_size 32
    
    # Or from code:
    from src.train import train_model
    train_model(data_dir="data", epochs=20)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_SCHEDULER, LR_PATIENCE, LR_FACTOR, LR_MIN,
    EARLY_STOPPING_PATIENCE, USE_AMP, MODEL_PATH, CHECKPOINT_PATH,
    LOG_INTERVAL, SAVE_BEST_METRIC, RUNS_DIR, set_seed, print_config
)
from src.dataset import get_dataloaders
from src.model_builder import build_model, save_model, count_parameters
from src.metrics import calculate_metrics, plot_confusion_matrix


class Trainer:
    """
    Trainer class for skin lesion classification model.
    
    Handles training loop, validation, checkpointing, and logging with
    automatic mixed precision and OOM handling.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
        
    Example:
        trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
        history = trainer.train(epochs=20)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = DEVICE,
        use_amp: bool = USE_AMP
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        
        # Initialize GradScaler for mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                acc = 100.0 * correct / total
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, float]:
        """
        Validate the model.
        
        Returns:
            Average loss, accuracy, and F1 score
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds)
        )
        
        return epoch_loss, epoch_acc, metrics['macro_f1']
    
    def train(
        self,
        epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        save_best_metric: str = SAVE_BEST_METRIC
    ) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            scheduler: Learning rate scheduler
            early_stopping_patience: Epochs to wait before early stopping
            save_best_metric: Metric to use for saving best model ('f1', 'accuracy', 'loss')
            
        Returns:
            Training history dictionary
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Batch Size: {self.train_loader.batch_size}")
        print(f"Train Batches: {len(self.train_loader)}")
        print(f"Val Batches: {len(self.val_loader)}")
        print("-" * 70)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate(epoch)
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Check if this is the best model
            if save_best_metric == 'f1':
                current_metric = val_f1
            elif save_best_metric == 'accuracy':
                current_metric = val_acc
            else:  # loss
                current_metric = -val_loss  # Negative because we want to maximize
            
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                # Save best model
                save_model(
                    model=self.model,
                    save_path=str(MODEL_PATH),
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    scaler=self.scaler
                )
                
                # Save checkpoint
                save_model(
                    model=self.model,
                    save_path=str(CHECKPOINT_PATH),
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    scaler=self.scaler
                )
                
                print(f"✓ New best model! {save_best_metric}: {current_metric:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{epochs} - {epoch_time:.1f}s")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"LR: {current_lr:.6f} | Best {save_best_metric}: {self.best_metric:.4f} (epoch {self.best_epoch+1})")
            print("-" * 70)
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        print(f"\nTraining completed!")
        print(f"Best {save_best_metric}: {self.best_metric:.4f} at epoch {self.best_epoch+1}")
        
        return self.history


def plot_training_curves(history: Dict, save_path: str = "results.png") -> None:
    """
    Plot training curves for loss and accuracy.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', linewidth=2)
    axes[1, 0].set_title('Validation F1 Score', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'orange', linewidth=2)
    axes[1, 1].set_title('Learning Rate', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def train_model(
    data_dir: str,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    single_folder: bool = False
) -> Dict:
    """
    Main training function.
    
    Args:
        data_dir: Path to data directory
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        single_folder: Whether data is in single folder (will create splits)
        
    Returns:
        Training history dictionary
        
    Example:
        history = train_model(
            data_dir="data",
            epochs=20,
            batch_size=32
        )
    """
    # Set seed for reproducibility
    set_seed()
    
    # Print configuration
    print_config()
    
    # Try to load data with automatic batch size reduction on OOM
    train_loader, val_loader, _ = None, None, None
    current_batch_size = batch_size
    
    while train_loader is None:
        try:
            print(f"\nAttempting to load data with batch_size={current_batch_size}...")
            train_loader, val_loader, _ = get_dataloaders(
                data_dir=data_dir,
                batch_size=current_batch_size,
                single_folder=single_folder
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch_size > 4:
                current_batch_size //= 2
                print(f"OOM error! Reducing batch size to {current_batch_size}")
                torch.cuda.empty_cache()
            else:
                raise e
    
    # Build model
    print("\nBuilding model...")
    model = build_model()
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    if LR_SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=LR_MIN
        )
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_lr=LR_MIN,
            verbose=True
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        use_amp=USE_AMP
    )
    
    # Train
    history = trainer.train(
        epochs=epochs,
        scheduler=scheduler
    )
    
    # Save training curves
    plot_training_curves(history)
    
    # Save training log
    log_path = RUNS_DIR / "training_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Training completed\n")
        f.write(f"Best F1: {trainer.best_metric:.4f} at epoch {trainer.best_epoch+1}\n")
        f.write(f"Final train acc: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final val acc: {history['val_acc'][-1]:.2f}%\n")
    
    # Save run config
    run_config = {
        'epochs': epochs,
        'batch_size': current_batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'best_f1': trainer.best_metric,
        'best_epoch': trainer.best_epoch + 1,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
    }
    
    config_path = RUNS_DIR / "last_run.json"
    with open(config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    
    print(f"\nTraining artifacts saved to {RUNS_DIR}")
    
    return history


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='Train Skin Lesion Classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help=f'Weight decay (default: {WEIGHT_DECAY})')
    parser.add_argument('--single_folder', action='store_true',
                        help='Data is in single folder (will create splits)')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        single_folder=args.single_folder
    )


if __name__ == "__main__":
    main()
