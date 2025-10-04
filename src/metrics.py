"""
Metrics module for model evaluation.

Provides functions to calculate and visualize classification metrics
including accuracy, precision, recall, F1-score, and confusion matrix.

Example usage:
    from src.metrics import calculate_metrics, plot_confusion_matrix
    
    metrics = calculate_metrics(y_true, y_pred)
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png")
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from src.config import CLASS_NAMES, METRICS_DIR


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary containing various metrics
        
    Example:
        metrics = calculate_metrics(y_true, y_pred)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': {}
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        }
    
    return metrics


def print_metrics(metrics: Dict, class_names: List[str] = CLASS_NAMES) -> None:
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        class_names: List of class names
        
    Example:
        metrics = calculate_metrics(y_true, y_pred)
        print_metrics(metrics)
    """
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print("-" * 70)
    
    # Per-class metrics
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for class_name in class_names:
        if class_name in metrics['per_class']:
            class_metrics = metrics['per_class'][class_name]
            print(f"{class_name:<15} "
                  f"{class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} "
                  f"{class_metrics['f1']:<12.4f} "
                  f"{class_metrics['support']:<10}")
    
    print("-" * 70)
    print(f"{'Macro Avg':<15} "
          f"{metrics['macro_precision']:<12.4f} "
          f"{metrics['macro_recall']:<12.4f} "
          f"{metrics['macro_f1']:<12.4f}")
    print(f"{'Weighted Avg':<15} "
          f"{metrics['weighted_precision']:<12.4f} "
          f"{metrics['weighted_recall']:<12.4f} "
          f"{metrics['weighted_f1']:<12.4f}")
    print("=" * 70 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot (if None, uses default)
        normalize: Whether to normalize values to [0, 1]
        
    Example:
        plot_confusion_matrix(y_true, y_pred, save_path="cm.png")
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = METRICS_DIR / "confusion_matrix.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> str:
    """
    Get detailed classification report as string.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report string
        
    Example:
        report = get_classification_report(y_true, y_pred)
        print(report)
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )


def calculate_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, float]:
    """
    Calculate accuracy for each class separately.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary mapping class names to accuracies
        
    Example:
        accuracies = calculate_per_class_accuracy(y_true, y_pred)
        for class_name, acc in accuracies.items():
            print(f"{class_name}: {acc:.2f}%")
    """
    accuracies = {}
    
    for i, class_name in enumerate(class_names):
        # Get indices for this class
        class_indices = y_true == i
        
        if class_indices.sum() > 0:
            class_accuracy = (y_pred[class_indices] == y_true[class_indices]).mean()
            accuracies[class_name] = class_accuracy * 100
        else:
            accuracies[class_name] = 0.0
    
    return accuracies


def plot_per_class_metrics(
    metrics: Dict,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None
) -> None:
    """
    Plot per-class precision, recall, and F1-score as bar chart.
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
        class_names: List of class names
        save_path: Path to save plot
        
    Example:
        metrics = calculate_metrics(y_true, y_pred)
        plot_per_class_metrics(metrics, save_path="per_class_metrics.png")
    """
    # Extract per-class metrics
    precisions = [metrics['per_class'][name]['precision'] for name in class_names]
    recalls = [metrics['per_class'][name]['recall'] for name in class_names]
    f1_scores = [metrics['per_class'][name]['f1'] for name in class_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    # Create bars
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = METRICS_DIR / "per_class_metrics.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Per-class metrics plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Test metrics module
    print("Testing metrics module...")
    
    # Create dummy predictions
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 3, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=20, replace=False)
    y_pred[error_indices] = np.random.randint(0, 3, 20)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print_metrics(metrics)
    
    # Plot confusion matrix
    try:
        plot_confusion_matrix(y_true, y_pred)
        plot_per_class_metrics(metrics)
        print("\nMetrics module test passed!")
    except Exception as e:
        print(f"Error plotting: {e}")
