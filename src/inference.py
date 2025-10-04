"""
Inference module for Skin Lesion Classifier.

Provides functions to load models and perform predictions on new images
with optional Grad-CAM visualization.

Example usage:
    from src.inference import predict, predict_from_path
    
    result = predict_from_path("image.jpg")
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
"""

from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import (
    MODEL_PATH, DEVICE, CLASS_NAMES, IMAGE_SIZE,
    IMAGENET_MEAN, IMAGENET_STD, CONFIDENCE_THRESHOLD
)
from src.model_builder import build_model, load_checkpoint
from src.dataset import get_val_transforms
from src.gradcam import GradCAM, create_gradcam_overlay
from src.utils import load_image


# Global model cache
_MODEL_CACHE = None


def load_model(model_path: str = str(MODEL_PATH), device: torch.device = DEVICE) -> torch.nn.Module:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
        
    Example:
        model = load_model("models/model.pt")
    """
    global _MODEL_CACHE
    
    # Return cached model if available
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    # Check if model file exists
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train a model first using: python -m src.train --data_dir data"
        )
    
    # Build and load model
    model = build_model()
    model = load_checkpoint(model, model_path, device=device)
    model.eval()
    
    # Cache model
    _MODEL_CACHE = model
    
    return model


def predict_from_tensor(
    image_tensor: torch.Tensor,
    model: Optional[torch.nn.Module] = None,
    return_probabilities: bool = True,
    generate_gradcam: bool = True
) -> Dict:
    """
    Make prediction from preprocessed image tensor.
    
    Args:
        image_tensor: Preprocessed image tensor (1, 3, H, W) or (3, H, W)
        model: Model to use (if None, loads default)
        return_probabilities: Whether to return class probabilities
        generate_gradcam: Whether to generate Grad-CAM heatmap
        
    Returns:
        Dictionary with prediction results
        
    Example:
        result = predict_from_tensor(image_tensor)
        print(f"Label: {result['label']}")
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Ensure tensor has batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    # Convert to numpy
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    probabilities_np = probabilities.cpu().numpy()[0]
    
    # Build result dictionary
    result = {
        'class_index': predicted_class,
        'label': CLASS_NAMES[predicted_class],
        'confidence': confidence,
    }
    
    if return_probabilities:
        result['probabilities'] = probabilities_np
        result['class_probabilities'] = {
            name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities_np)
        }
    
    # Generate Grad-CAM if requested
    if generate_gradcam:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate_heatmap(image_tensor, target_class=predicted_class)
        result['gradcam_heatmap'] = heatmap
    
    return result


def predict_from_array(
    image_array: np.ndarray,
    model: Optional[torch.nn.Module] = None,
    return_probabilities: bool = True,
    generate_gradcam: bool = True
) -> Dict:
    """
    Make prediction from numpy image array.
    
    Args:
        image_array: Image as numpy array (H, W, 3) in RGB format
        model: Model to use (if None, loads default)
        return_probabilities: Whether to return class probabilities
        generate_gradcam: Whether to generate Grad-CAM heatmap
        
    Returns:
        Dictionary with prediction results and optional Grad-CAM overlay
        
    Example:
        result = predict_from_array(image_array)
        overlay = result['gradcam_overlay']
    """
    # Get transform
    transform = get_val_transforms()
    
    # Transform image
    transformed = transform(image=image_array)
    image_tensor = transformed['image']
    
    # Make prediction
    result = predict_from_tensor(
        image_tensor=image_tensor,
        model=model,
        return_probabilities=return_probabilities,
        generate_gradcam=generate_gradcam
    )
    
    # Create Grad-CAM overlay if heatmap was generated
    if generate_gradcam and 'gradcam_heatmap' in result:
        heatmap = result['gradcam_heatmap']
        overlay = create_gradcam_overlay(image_array, heatmap)
        result['gradcam_overlay'] = overlay
        result['original_image'] = image_array
    
    return result


def predict_from_path(
    image_path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    return_probabilities: bool = True,
    generate_gradcam: bool = True
) -> Dict:
    """
    Make prediction from image file path.
    
    Args:
        image_path: Path to image file
        model: Model to use (if None, loads default)
        return_probabilities: Whether to return class probabilities
        generate_gradcam: Whether to generate Grad-CAM heatmap
        
    Returns:
        Dictionary with prediction results
        
    Example:
        result = predict_from_path("skin_lesion.jpg")
        print(f"Prediction: {result['label']} ({result['confidence']:.2%})")
        
        # Display Grad-CAM overlay
        import matplotlib.pyplot as plt
        plt.imshow(result['gradcam_overlay'])
        plt.show()
    """
    # Load image
    image_array = load_image(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    
    # Make prediction
    result = predict_from_array(
        image_array=image_array,
        model=model,
        return_probabilities=return_probabilities,
        generate_gradcam=generate_gradcam
    )
    
    result['image_path'] = str(image_path)
    
    return result


def predict_batch(
    image_paths: list,
    model: Optional[torch.nn.Module] = None,
    batch_size: int = 32,
    generate_gradcam: bool = False
) -> list:
    """
    Make predictions on a batch of images.
    
    Args:
        image_paths: List of image file paths
        model: Model to use (if None, loads default)
        batch_size: Batch size for processing
        generate_gradcam: Whether to generate Grad-CAM (slower)
        
    Returns:
        List of prediction dictionaries
        
    Example:
        results = predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
        for result in results:
            print(f"{result['image_path']}: {result['label']}")
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    results = []
    transform = get_val_transforms()
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and transform images
        batch_tensors = []
        batch_arrays = []
        
        for img_path in batch_paths:
            img_array = load_image(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            transformed = transform(image=img_array)
            batch_tensors.append(transformed['image'])
            batch_arrays.append(img_array)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tensors).to(DEVICE)
        
        # Make predictions
        with torch.no_grad():
            logits = model(batch_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidences, predicted_classes = torch.max(probabilities, dim=1)
        
        # Convert to numpy
        predicted_classes = predicted_classes.cpu().numpy()
        confidences = confidences.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        
        # Build results
        for j, img_path in enumerate(batch_paths):
            result = {
                'image_path': str(img_path),
                'class_index': int(predicted_classes[j]),
                'label': CLASS_NAMES[predicted_classes[j]],
                'confidence': float(confidences[j]),
                'probabilities': probabilities_np[j],
                'class_probabilities': {
                    name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities_np[j])
                }
            }
            
            # Generate Grad-CAM if requested (slower)
            if generate_gradcam:
                gradcam = GradCAM(model)
                heatmap = gradcam.generate_heatmap(
                    batch_tensor[j:j+1],
                    target_class=predicted_classes[j]
                )
                overlay = create_gradcam_overlay(batch_arrays[j], heatmap)
                result['gradcam_heatmap'] = heatmap
                result['gradcam_overlay'] = overlay
            
            results.append(result)
    
    return results


def is_confident(result: Dict, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """
    Check if prediction confidence is above threshold.
    
    Args:
        result: Prediction result dictionary
        threshold: Confidence threshold
        
    Returns:
        True if confidence >= threshold
        
    Example:
        result = predict_from_path("image.jpg")
        if is_confident(result, threshold=0.7):
            print("High confidence prediction")
    """
    return result['confidence'] >= threshold


def get_top_k_predictions(
    result: Dict,
    k: int = 3
) -> list:
    """
    Get top-k predictions with probabilities.
    
    Args:
        result: Prediction result dictionary
        k: Number of top predictions to return
        
    Returns:
        List of (class_name, probability) tuples
        
    Example:
        result = predict_from_path("image.jpg")
        top_3 = get_top_k_predictions(result, k=3)
        for class_name, prob in top_3:
            print(f"{class_name}: {prob:.2%}")
    """
    if 'probabilities' not in result:
        raise ValueError("Result does not contain probabilities")
    
    probabilities = result['probabilities']
    top_k_indices = np.argsort(probabilities)[-k:][::-1]
    
    top_k = [(CLASS_NAMES[i], float(probabilities[i])) for i in top_k_indices]
    
    return top_k


if __name__ == "__main__":
    # Test inference module
    print("Testing inference module...")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Model not found at {MODEL_PATH}")
        print("Please train a model first using: python -m src.train --data_dir data")
    else:
        try:
            # Load model
            print("Loading model...")
            model = load_model()
            print("Model loaded successfully!")
            
            # Create dummy image
            print("\nTesting with dummy image...")
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Make prediction
            result = predict_from_array(dummy_image, model=model)
            
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"All probabilities:")
            for class_name, prob in result['class_probabilities'].items():
                print(f"  {class_name}: {prob:.2%}")
            
            if 'gradcam_overlay' in result:
                print(f"Grad-CAM overlay shape: {result['gradcam_overlay'].shape}")
            
            print("\nInference module test passed!")
            
        except Exception as e:
            print(f"Error testing inference: {e}")
            import traceback
            traceback.print_exc()
