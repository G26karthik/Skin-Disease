"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

This module provides explainable AI visualization for the skin lesion classifier
by highlighting which regions of the image the model focuses on for predictions.

Example usage:
    from src.gradcam import GradCAM, create_gradcam_overlay
    
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_heatmap(image, target_class=1)
    overlay = create_gradcam_overlay(original_image, heatmap)
"""

from typing import Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.config import GRADCAM_ALPHA, DEVICE


class GradCAM:
    """
    Grad-CAM implementation for visualizing CNN decisions.
    
    Grad-CAM uses gradients of the target class flowing into the final
    convolutional layer to highlight important regions in the image.
    
    Args:
        model: PyTorch model
        target_layer: Target convolutional layer (if None, uses last conv layer)
        
    Example:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate_heatmap(image_tensor, target_class=1)
        # heatmap is a numpy array of shape (H, W) with values in [0, 1]
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        # Use provided target layer or get it from model
        if target_layer is None:
            self.target_layer = model.get_gradcam_target_layer()
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image: Input tensor of shape (1, 3, H, W) or (3, H, W)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Heatmap as numpy array of shape (H, W) with values in [0, 1]
            
        Example:
            image_tensor = transform(image).unsqueeze(0).to(device)
            heatmap = gradcam.generate_heatmap(image_tensor, target_class=1)
        """
        # Ensure image has batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(DEVICE)
        image.requires_grad = True
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        heatmap = torch.zeros(activations.shape[1:], device=DEVICE)
        for i, weight in enumerate(weights):
            heatmap += weight * activations[i]
        
        # Apply ReLU (only positive contributions)
        heatmap = F.relu(heatmap)
        
        # Normalize to [0, 1]
        heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def __call__(self, image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Allow GradCAM to be called as a function."""
        return self.generate_heatmap(image, target_class)


def create_gradcam_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = GRADCAM_ALPHA,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Create an overlay of the Grad-CAM heatmap on the original image.
    
    Args:
        original_image: Original image as numpy array (H, W, 3) in RGB, values [0, 255]
        heatmap: Grad-CAM heatmap (H', W') with values [0, 1]
        alpha: Transparency of heatmap overlay (0=invisible, 1=opaque)
        colormap: OpenCV colormap to apply to heatmap
        
    Returns:
        Overlay image as numpy array (H, W, 3) in RGB, values [0, 255]
        
    Example:
        overlay = create_gradcam_overlay(
            original_image=image_array,
            heatmap=heatmap,
            alpha=0.4
        )
    """
    # Ensure original image is RGB uint8
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Convert BGR to RGB (OpenCV uses BGR)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def create_gradcam_visualization(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = GRADCAM_ALPHA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create both overlay and standalone heatmap visualizations.
    
    Args:
        original_image: Original image as numpy array (H, W, 3) in RGB
        heatmap: Grad-CAM heatmap (H', W') with values [0, 1]
        alpha: Transparency for overlay
        
    Returns:
        Tuple of (overlay, heatmap_colored) both as numpy arrays
        
    Example:
        overlay, heatmap_img = create_gradcam_visualization(image, heatmap)
    """
    # Create overlay
    overlay = create_gradcam_overlay(original_image, heatmap, alpha)
    
    # Create standalone colored heatmap
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return overlay, heatmap_colored


def gradcam_from_image_path(
    model: nn.Module,
    image_path: str,
    transform: callable,
    target_class: Optional[int] = None,
    alpha: float = GRADCAM_ALPHA
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    End-to-end Grad-CAM generation from image path.
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        transform: Transform function to preprocess image
        target_class: Target class (if None, uses prediction)
        alpha: Overlay transparency
        
    Returns:
        Tuple of (overlay, heatmap_colored, predicted_class)
        
    Example:
        overlay, heatmap, pred = gradcam_from_image_path(
            model=model,
            image_path="image.jpg",
            transform=transform
        )
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Transform for model
    image_tensor = transform(image=image_np)['image']
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Get prediction if target_class not specified
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor)
            target_class = output.argmax(dim=1).item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam.generate_heatmap(image_tensor, target_class)
    
    # Create visualizations
    overlay, heatmap_colored = create_gradcam_visualization(image_np, heatmap, alpha)
    
    return overlay, heatmap_colored, target_class


def batch_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    target_classes: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Generate Grad-CAM heatmaps for a batch of images.
    
    Args:
        model: PyTorch model
        images: Batch of images (N, 3, H, W)
        target_classes: Target classes for each image (if None, uses predictions)
        
    Returns:
        Array of heatmaps (N, H', W')
        
    Example:
        heatmaps = batch_gradcam(model, image_batch)
    """
    gradcam = GradCAM(model)
    heatmaps = []
    
    for i in range(images.size(0)):
        image = images[i:i+1]
        target_class = target_classes[i].item() if target_classes is not None else None
        heatmap = gradcam.generate_heatmap(image, target_class)
        heatmaps.append(heatmap)
    
    return np.array(heatmaps)


if __name__ == "__main__":
    # Test Grad-CAM implementation
    print("Testing Grad-CAM module...")
    
    try:
        from src.model_builder import build_model
        from src.dataset import get_val_transforms
        
        # Build model
        model = build_model(pretrained=False)
        model.eval()
        
        # Create dummy image
        dummy_image = torch.randn(1, 3, 224, 224).to(DEVICE)
        
        # Generate Grad-CAM
        gradcam = GradCAM(model)
        heatmap = gradcam.generate_heatmap(dummy_image, target_class=0)
        
        print(f"Heatmap shape: {heatmap.shape}")
        print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        print(f"Heatmap mean: {heatmap.mean():.3f}")
        
        # Test overlay creation
        dummy_image_np = (torch.rand(224, 224, 3).numpy() * 255).astype(np.uint8)
        overlay = create_gradcam_overlay(dummy_image_np, heatmap)
        
        print(f"Overlay shape: {overlay.shape}")
        print(f"Overlay dtype: {overlay.dtype}")
        
        print("\nGrad-CAM module test passed!")
        
    except Exception as e:
        print(f"Error testing Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
