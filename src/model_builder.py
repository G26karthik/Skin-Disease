"""
Model builder module for Skin Lesion Classification.

This module provides functions to build and configure the EfficientNet_B0
model with transfer learning for 3-class classification.

Example usage:
    from src.model_builder import build_model
    
    model = build_model(num_classes=3, pretrained=True)
    model = model.to(device)
    
    output = model(images)  # Shape: (batch_size, 3)
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.config import NUM_CLASSES, PRETRAINED, DROPOUT_RATE, DEVICE


class SkinLesionClassifier(nn.Module):
    """
    EfficientNet_B0-based classifier for skin lesion classification.
    
    Uses transfer learning with a custom classification head.
    
    Args:
        num_classes: Number of output classes (default: 3)
        pretrained: Whether to use pretrained ImageNet weights
        dropout_rate: Dropout probability for regularization
        
    Example:
        model = SkinLesionClassifier(num_classes=3, pretrained=True)
        output = model(images)  # Shape: (batch_size, 3)
    """
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        dropout_rate: float = DROPOUT_RATE
    ):
        super(SkinLesionClassifier, self).__init__()
        
        # Load pretrained EfficientNet_B0
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        # Get number of input features for the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # Initialize weights of the new classifier
        self._initialize_weights()
        
        self.num_classes = num_classes
    
    def _initialize_weights(self):
        """Initialize weights of the classifier head."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps from the last convolutional layer.
        
        Useful for Grad-CAM visualization.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Feature maps of shape (batch_size, 1280, 7, 7)
        """
        return self.backbone.features(x)
    
    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Get the target layer for Grad-CAM.
        
        Returns:
            The last convolutional layer
        """
        # For EfficientNet_B0, the last conv layer is in features
        return self.backbone.features[-1]


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED,
    dropout_rate: float = DROPOUT_RATE,
    device: torch.device = DEVICE
) -> SkinLesionClassifier:
    """
    Build and initialize the skin lesion classifier model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights
        dropout_rate: Dropout probability
        device: Device to move the model to
        
    Returns:
        Initialized model on the specified device
        
    Example:
        model = build_model(num_classes=3, pretrained=True)
        print(f"Model has {count_parameters(model):,} parameters")
    """
    model = SkinLesionClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    model = model.to(device)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
        
    Example:
        model = build_model()
        print(f"Trainable parameters: {count_parameters(model):,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: SkinLesionClassifier, freeze: bool = True) -> SkinLesionClassifier:
    """
    Freeze or unfreeze the backbone layers for transfer learning.
    
    Useful for two-stage training: first train only the head,
    then fine-tune the entire network.
    
    Args:
        model: The classifier model
        freeze: Whether to freeze (True) or unfreeze (False) backbone
        
    Returns:
        Modified model
        
    Example:
        # Stage 1: Train only classifier head
        model = freeze_backbone(model, freeze=True)
        # ... train for a few epochs ...
        
        # Stage 2: Fine-tune entire network
        model = freeze_backbone(model, freeze=False)
        # ... continue training ...
    """
    for param in model.backbone.features.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Backbone {status}. Trainable parameters: {count_parameters(model):,}")
    
    return model


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Model summary string
        
    Example:
        model = build_model()
        print(get_model_summary(model))
    """
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("MODEL SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Model type: {type(model).__name__}")
    summary_lines.append(f"Number of classes: {model.num_classes}")
    summary_lines.append(f"Total parameters: {count_parameters(model):,}")
    
    # Count parameters by component
    backbone_params = sum(p.numel() for p in model.backbone.features.parameters())
    classifier_params = sum(p.numel() for p in model.backbone.classifier.parameters())
    
    summary_lines.append(f"Backbone parameters: {backbone_params:,}")
    summary_lines.append(f"Classifier parameters: {classifier_params:,}")
    summary_lines.append(f"Input size: {input_size}")
    
    # Test forward pass to get output size
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
        output = model(dummy_input)
        summary_lines.append(f"Output size: {tuple(output.shape)}")
    
    summary_lines.append("=" * 70)
    
    return "\n".join(summary_lines)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = DEVICE,
    strict: bool = True
) -> nn.Module:
    """
    Load model weights from a checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load the model on
        strict: Whether to strictly enforce key matching
        
    Returns:
        Model with loaded weights
        
    Example:
        model = build_model()
        model = load_checkpoint(model, "models/checkpoint.pth")
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint and state_dict only
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint, strict=strict)
    
    model = model.to(device)
    model.eval()
    
    return model


def save_model(
    model: nn.Module,
    save_path: str,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = None,
    best_metric: float = None,
    scaler: torch.cuda.amp.GradScaler = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional)
        best_metric: Best metric value (optional)
        scaler: GradScaler for mixed precision (optional)
        
    Example:
        save_model(
            model=model,
            save_path="models/checkpoint.pth",
            optimizer=optimizer,
            epoch=10,
            best_metric=0.95
        )
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Test model building
    print("Testing model builder...")
    
    try:
        # Build model
        model = build_model(num_classes=3, pretrained=False)
        print(get_model_summary(model))
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
        output = model(dummy_input)
        print(f"\nTest forward pass:")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test feature extraction for Grad-CAM
        features = model.get_features(dummy_input)
        print(f"Feature map shape: {features.shape}")
        
        print("\nModel builder test passed!")
        
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
