# AI Project-Based Learning (PBL) Report

## Visual Snapshots

- **System Architecture:** ![Architecture](docs/images/architecture.png)
- **Class Distribution:** ![Class Distribution](docs/images/class_distribution.png)
- **Grad-CAM Explainability:** ![Grad-CAM Montage](docs/images/gradcam_montage.png)

---

## 1. Introduction

This project, titled **AI-Assisted Dermatological Decision Support System (DSS)**, aims to develop a research-grade artificial intelligence tool for automated classification of dermatoscopic skin lesion images. The primary problem addressed is the early and accurate detection of skin cancer and other dermatological conditions using deep learning, with a focus on clinical interpretability and reliability. The system is designed to classify images into seven clinically relevant categories (e.g., Melanoma, Benign Nevus) and provide visual explanations for its predictions.

The motivation for this project stems from the need for scalable, accurate, and explainable AI tools in medical diagnostics, particularly in dermatology where early detection of malignant lesions can significantly improve patient outcomes. The expected outcome is a robust, production-ready AI system that demonstrates high accuracy, strong calibration, and transparent decision-making suitable for academic research, portfolio demonstration, and future clinical translation.

## 2. System Design

### System Architecture
The DSS project follows a modular AI pipeline:
- **Data Ingestion**: Raw dermatoscopic images are loaded from the HAM10000 dataset, organized into train, validation, and test splits with strict lesion-level partitioning to prevent data leakage.
- **Preprocessing**: Images are resized to 224x224 pixels, normalized using ImageNet statistics, and optionally augmented (flip, rotate, color jitter) to improve generalization.
- **Model Training**: The core model is EfficientNet_B0, fine-tuned on the HAM10000 dataset using mixed precision (AMP) for speed and memory efficiency. Training is performed for 30 epochs with early stopping and checkpointing.
- **Evaluation**: Model performance is assessed using accuracy, macro F1-score, calibration metrics (ECE, Brier score), and per-class metrics. Confusion matrices and reliability diagrams are generated for interpretability.
- **Deployment**: The trained model is deployed via a Streamlit web application, allowing users to upload images and receive instant predictions with Grad-CAM visual explanations.

### Modules
- **Data Preparation**: Scripts for dataset preprocessing, stratified splitting, and augmentation.
- **Model Building**: Modular code for constructing EfficientNet_B0 with a custom classification head.
- **Training & Hyperparameter Tuning**: Configurable training loop supporting AMP, learning rate scheduling, and batch size adjustment.
- **Testing & Evaluation**: Automated scripts for generating metrics, confusion matrices, and calibration plots.
- **Explainability**: Grad-CAM implementation for visualizing model attention.
- **Deployment**: Streamlit app for real-time inference and user interaction.

### Backend Design
- **Storage**: Model weights, metrics, and visualizations are stored in organized directories (`models/`, `runs/`, `docs/images/`).
- **APIs**: The Streamlit app serves as the primary interface; no external cloud APIs are used, ensuring privacy and local processing.

### Architecture Flow (Textual Description)
The system processes an input image as follows:
1. Image is loaded and preprocessed.
2. EfficientNet_B0 backbone extracts features and produces class logits.
3. Softmax converts logits to probabilities; calibration metrics are computed.
4. Grad-CAM generates a heatmap overlay for interpretability.
5. Output includes predicted class, probabilities, heatmap, and latency metrics.

## 3. Implementation

### Tools, Frameworks, and Environment
- **Programming Language**: Python 3.10+
- **Deep Learning Framework**: PyTorch 2.7.1+cu118
- **Vision Model**: torchvision EfficientNet_B0
- **Data Handling**: NumPy, Pandas, Albumentations
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit
- **Testing**: pytest

### Dataset
- **Source**: HAM10000 (Kaggle, 10,015 dermatoscopic images)
- **Type**: Multiclass image classification (7 skin lesion categories)
- **Size**: 17,036 images (train+val), 2,994 images (test)
- **Features**: RGB images, 224x224 pixels
- **Preprocessing**: Resize, normalization (ImageNet stats), optional augmentation
- **Split**: Stratified by lesion, preventing data leakage

### Model Implementation
- **Type**: Convolutional Neural Network (CNN) — EfficientNet_B0 backbone
- **Layers**: Pretrained EfficientNet_B0, custom linear head (1280→7), dropout (0.3)
- **Activation**: ReLU (hidden), Softmax (output)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Hyperparameters**: Learning rate 1e-3, batch size 32, epochs 30, mixed precision (AMP)

#### Sample Code Snippet: Model Creation
```python
import torch.nn as nn
from torchvision.models import efficientnet_b0

def build_model(num_classes=7):
    base = efficientnet_b0(pretrained=True)
    in_features = base.classifier[1].in_features
    base.classifier[1] = nn.Linear(in_features, num_classes)
    return base
```

#### Sample Code Snippet: Training Loop
```python
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    # Validation and metrics...
```

#### Sample Code Snippet: Evaluation & Inference
```python
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(1)
        # Collect metrics...
```

#### Sample Code Snippet: Data Preprocessing
```python
from torchvision import transforms
from PIL import Image

def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)
```

#### Sample Code Snippet: Grad-CAM Visualization
```python
import torch
import numpy as np
from src.gradcam import GradCAM

def generate_gradcam(model, image_tensor, target_class):
    gradcam = GradCAM(model, target_layer='features')
    mask = gradcam(image_tensor.unsqueeze(0), target_class)
    return np.uint8(255 * mask)
```

#### Sample Code Snippet: Streamlit Inference UI
```python
import streamlit as st
from PIL import Image
import numpy as np

st.header("Skin Lesion Classifier")
file = st.file_uploader("Upload Image", type=['jpg', 'png'])
if file:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image")
    # Preprocess and predict...
```

### Result Interpretations
- **Accuracy & Loss Curves**: Training and validation accuracy steadily increased over 30 epochs, with validation accuracy plateauing near 85.8%. Loss decreased consistently, indicating stable learning and no overfitting.
- **Confusion Matrix**: The normalized confusion matrix shows strong diagonal dominance, with highest accuracy for Melanocytic nevi and lower recall for Melanoma, reflecting clinical challenges in minority class detection.
- **Per-Class Metrics**: Macro F1-score reached 0.7512, with Melanocytic nevi achieving 0.9306 F1 and Melanoma 0.6667 F1. Precision and recall varied by class, with minority classes showing lower recall.
- **Calibration Metrics**: Expected Calibration Error (ECE) was 0.0450, indicating well-calibrated probability outputs. Brier score was 0.2764, supporting reliability.
- **Latency & Throughput**: Inference latency averaged 6.29ms per image (RTX 4060 GPU), with throughput of 158.89 images/sec, suitable for real-time clinical use.

#### Metric Significance
- **Accuracy**: High overall accuracy (85.77%) demonstrates robust classification performance.
- **Macro F1**: Balanced performance across all classes, important for clinical safety.
- **Confusion Matrix**: Reveals strengths and weaknesses in class-wise predictions, guiding future improvements.
- **Calibration**: Reliable probability estimates are critical for medical decision support.
- **Latency**: Fast inference enables practical deployment in clinical settings.

## 4. Conclusion

The AI-Assisted Dermatological Decision Support System successfully achieved its objectives of building a robust, explainable, and high-performance skin lesion classifier. The model reached 85.77% test accuracy and a macro F1-score of 0.7512, demonstrating strong generalization and balanced class-wise performance. Calibration metrics confirmed reliable probability outputs, and real-time inference was validated on consumer-grade GPU hardware.

Key findings include:
- EfficientNet_B0 is effective for medical image classification, balancing accuracy and computational efficiency.
- Grad-CAM visualizations provide meaningful interpretability, supporting clinical trust.
- Stratified lesion-level splitting and careful preprocessing prevented data leakage and overfitting.
- Automated evaluation and visualization pipelines ensured reproducibility and transparency.

Limitations encountered:
- Minority class recall (e.g., Melanoma) remains lower than ideal, reflecting real-world clinical challenges.
- Dataset diversity is limited to HAM10000; external validation is needed for broader generalization.
- Advanced uncertainty quantification and fairness metrics are planned for future work.

Lessons learned:
- Calibration and interpretability are essential for medical AI adoption.
- Modular, reproducible codebases accelerate research and deployment.

Future improvements:
- Incorporate additional datasets for external validation.
- Expand explainability with SHAP, LIME, and counterfactuals.
- Integrate uncertainty estimation and fairness analysis.
- Prepare for clinical deployment with containerization and monitoring.

### Performance Visualizations

- **Per-Class Metrics:** ![Per-Class Metrics](docs/images/per_class_metrics.png)
- **Latency Profile:** ![Latency](docs/images/latency.png)
- **Project Roadmap:** ![Roadmap](docs/images/roadmap.png)

---

## 5. References

- Tschandl P. et al. HAM10000 Dataset. Scientific Data (2018).
- Tan M., Le Q. EfficientNet: Rethinking Model Scaling for CNNs. ICML (2019).
- Selvaraju R.R. et al. Grad-CAM: Visual Explanations from Deep Networks. ICCV (2017).
- Guo C. et al. On Calibration of Modern Neural Networks. ICML (2017).
- He K. et al. Deep Residual Learning for Image Recognition. CVPR (2016).
- Dosovitskiy A. et al. An Image Is Worth 16x16 Words (ViT). ICLR (2021).
- PyTorch (https://pytorch.org/)
- torchvision (https://pytorch.org/vision/stable/index.html)
- Streamlit (https://streamlit.io/)
- Kaggle HAM10000 Dataset (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Albumentations (https://albumentations.ai/)
- Matplotlib (https://matplotlib.org/)
- Seaborn (https://seaborn.pydata.org/)
- NumPy (https://numpy.org/)
- Pandas (https://pandas.pydata.org/)
- pytest (https://docs.pytest.org/)

### Additional Visuals

- **Confusion Matrix (Test Set):** ![Confusion Matrix](docs/images/confusion_matrix.png)
- **Reliability Diagram (Calibration):** ![Reliability](docs/images/reliability.png)
- **Training Curves:** ![Training Curves](docs/images/training_curves.png)

---