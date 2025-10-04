# 🔬 AI-Powered Skin Lesion Classifier with Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

A production-quality deep learning system for classifying skin lesions using the **HAM10000 dataset** (10,015 dermatoscopic images across 7 diagnostic categories). Features explainable AI (Grad-CAM) visualization, PyTorch/EfficientNet_B0 architecture, and GPU-accelerated mixed precision training.

**⚠️ MEDICAL DISCLAIMER**: This tool is for educational and research purposes only. It is NOT a substitute for professional medical diagnosis. Always consult a qualified dermatologist or healthcare provider.

---

## 🎯 Features

- **HAM10000 Dataset**: 10,015 dermatoscopic images across 7 skin lesion types
  - Melanocytic nevi (nv) - benign moles
  - Melanoma (mel) - malignant
  - Benign keratosis (bkl) - benign
  - Basal cell carcinoma (bcc) - malignant
  - Actinic keratoses (akiec) - pre-cancerous
  - Vascular lesions (vasc) - benign
  - Dermatofibroma (df) - benign
- **Transfer Learning**: EfficientNet_B0 pretrained on ImageNet, fine-tuned for dermatoscopic classification
- **Explainable AI**: Grad-CAM heatmaps showing model attention regions
- **GPU Optimized**: Mixed precision training (AMP) for 2-3x speedup on NVIDIA GPUs (CUDA 11.8)
- **Production Ready**: 
  - Automatic OOM handling with batch size reduction
  - Early stopping and learning rate scheduling
  - Comprehensive metrics (accuracy, F1, confusion matrix)
  - Model checkpointing and reproducible training
- **Interactive Demo**: Streamlit web app with real-time inference and visualization
- **Well-Tested**: Unit tests for data loading and inference pipelines

---

## 📊 Project Structure

```
ai-skin-lesion-xai/
├── src/
│   ├── config.py           # Configuration and hyperparameters
│   ├── dataset.py          # PyTorch Dataset with augmentation
│   ├── model_builder.py    # EfficientNet_B0 model builder
│   ├── train.py            # Training pipeline with AMP
│   ├── inference.py        # Prediction and inference utilities
│   ├── gradcam.py          # Grad-CAM implementation
│   ├── metrics.py          # Evaluation metrics
│   └── utils.py            # Helper functions
├── scripts/
│   ├── preprocess_ham10000.py  # HAM10000 dataset preprocessing
│   └── preprocess_isic.py      # Legacy ISIC preprocessing (optional)
├── tests/
│   ├── test_dataset.py     # Dataset tests
│   └── test_inference.py   # Inference tests
├── app.py                  # Streamlit demo application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd ai-skin-lesion-xai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get HAM10000 Dataset

**Dataset**: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

#### Setup Kaggle API:

```bash
# Install Kaggle CLI
pip install kaggle

# Place your kaggle.json in the correct location:
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Download from: https://www.kaggle.com/settings -> API -> Create New API Token
```

#### Download and Preprocess:

```bash
# Download HAM10000 dataset (5.2GB)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw --unzip

# Preprocess into train/val/test splits (70%/15%/15%)
python scripts/preprocess_ham10000.py

# Output: data/ham10000/
#   ├── train/  (7,020 images)
#   ├── val/    (1,498 images)
#   └── test/   (1,497 images)
```

**Dataset Stats**:
- Total: 10,015 dermatoscopic images
- Resolution: 600x450 to 1024x768 (resized to 224x224)
- Format: JPG
- Classes: 7 (nv, mel, bkl, bcc, akiec, vasc, df)
- Train/Val/Test: 70%/15%/15% stratified split

### 3. Train the Model

```bash
# Train with default settings (optimized for NVIDIA RTX GPUs)
python -m src.train --data_dir data/ham10000 --epochs 30 --batch_size 32

# Train with custom settings
python -m src.train \
    --data_dir data/ham10000 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-4
```

**GPU Recommendations:**
- Recommended batch size: 16-32
- Mixed precision enabled automatically
- Training time: ~30-45 min for 20 epochs (depends on dataset size)

### 4. Run Inference

```bash
# Test inference on a single image
python -c "
from src.inference import predict_from_path
result = predict_from_path('path/to/image.jpg')
print(f'Prediction: {result[\"label\"]} ({result[\"confidence\"]:.2%})')
"
```

### 5. Launch Demo App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🏗️ Architecture

```
                    Input Image (224x224x3)
                            ↓
                  [Data Augmentation]
                (Rotation, Flip, Color Jitter)
                            ↓
                  [EfficientNet_B0 Backbone]
            (Pretrained on ImageNet, ~5M params)
                            ↓
                [Global Average Pooling]
                            ↓
                    [Dropout (p=0.3)]
                            ↓
                    [Linear (1280→7)]
                            ↓
                      [Softmax]
                            ↓
    [akiec | bcc | bkl | df | mel | nv | vasc]
                            ↓
                    [Grad-CAM Layer]
              (Highlights important regions)
```

---

## 📈 Training Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | EfficientNet_B0 | Pretrained on ImageNet |
| Input Size | 224×224 | Standard for EfficientNet_B0 |
| Batch Size | 32 (auto-adjusted) | Reduced to 16 if GPU memory < 8GB |
| Learning Rate | 1e-4 | With AdamW optimizer |
| Epochs | 20 | With early stopping (patience=5) |
| Augmentation | Yes | Rotation, flip, color jitter, crop |
| Mixed Precision | Auto | Enabled on CUDA devices |
| Scheduler | Cosine Annealing | Reduces LR to 1e-6 |

### Performance Metrics

After training, the model outputs:
- Training/validation loss and accuracy curves
- Per-class precision, recall, F1-score
- Confusion matrix
- Macro F1-score (primary metric)

**Expected Performance** (on ISIC 2018/2020):
- Accuracy: 75-85%
- Macro F1: 0.70-0.80
- Model size: ~20MB

*Note: Actual performance depends on dataset size and quality*

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_dataset.py -v
```

---

## 🌐 Deployment

### Deploy to Hugging Face Spaces

1. Create a new Space on https://huggingface.co/spaces
2. Choose Streamlit as the SDK
3. Upload these files:
   ```
   app.py
   requirements.txt
   models/model.pt  (your trained model)
   src/ (entire directory)
   ```
4. Add `python_version: 3.10` to your Space settings
5. Your app will be live at `https://huggingface.co/spaces/<username>/<space-name>`

### Deploy to Cloud (AWS/Azure/GCP)

1. Containerize with Docker:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

2. Build and push:

```bash
docker build -t skin-lesion-classifier .
docker push <your-registry>/skin-lesion-classifier
```

3. Deploy to your cloud provider's container service

---

## 🔧 Configuration

Key settings in `src/config.py`:

```python
# Hardware
DEVICE = "cuda"  # Auto-detected
USE_AMP = True   # Mixed precision

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# Data
IMAGE_SIZE = 224
NUM_CLASSES = 7
CLASS_NAMES = [
    "Melanocytic_nevi",      # nv - benign moles
    "Melanoma",              # mel - malignant
    "Benign_keratosis",      # bkl - benign lesions
    "Basal_cell_carcinoma",  # bcc - malignant
    "Actinic_keratoses",     # akiec - pre-cancerous
    "Vascular_lesions",      # vasc - benign
    "Dermatofibroma"         # df - benign
]

# Augmentation
AUGMENTATION_CONFIG = {
    "rotation_limit": 20,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    ...
}
```

---

## 📊 Results Visualization

Training outputs:
- `results.png`: Loss and accuracy curves
- `metrics/confusion_matrix.png`: Confusion matrix
- `runs/training_log.txt`: Text log
- `runs/last_run.json`: Hyperparameters and metrics

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Dataset & Citation

### HAM10000 Dataset

This project uses the **HAM10000** (Human Against Machine with 10000 training images) dataset:

- **Source**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Size**: 10,015 dermatoscopic images
- **Classes**: 7 diagnostic categories
- **Resolution**: 600×450 pixels (resized to 224×224)
- **Format**: JPEG
- **License**: CC BY-NC 4.0 (free for academic/research use)

**Citation:**
```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific Data},
  volume={5},
  pages={180161},
  year={2018},
  publisher={Nature Publishing Group}
}
```

**Dataset Distribution:**
- Melanocytic nevi (nv): 6,705 images (67%)
- Melanoma (mel): 1,113 images (11%)
- Benign keratosis (bkl): 1,099 images (11%)
- Basal cell carcinoma (bcc): 514 images (5%)
- Actinic keratoses (akiec): 327 images (3%)
- Vascular lesions (vasc): 142 images (1.4%)
- Dermatofibroma (df): 115 images (1.1%)

---

## 🔐 Ethics & Limitations

### Ethical Considerations

- **Not a Diagnostic Tool**: This system is intended for educational and research purposes only
- **Clinical Oversight Required**: All predictions should be reviewed by qualified medical professionals
- **Bias Awareness**: Model performance may vary across different skin types and demographics
- **Data Privacy**: Follow HIPAA/GDPR guidelines if using with real patient data

### Known Limitations

- Model trained on publicly available datasets may not generalize to all populations
- Performance depends on image quality and lighting conditions
- Cannot replace dermatoscope examination or biopsy
- May have reduced accuracy on rare lesion types

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**G. Karthik Koundinya**
- Full-stack developer with ML expertise
- Specialized in computer vision and medical imaging applications

---

## 🙏 Acknowledgments

- ISIC for providing the dataset
- PyTorch and Hugging Face communities
- EfficientNet authors for the architecture
- Grad-CAM authors for the visualization technique

---

## 📞 Support

For questions or issues:
1. Check existing GitHub Issues
2. Create a new Issue with detailed description
3. Include error logs and environment details

---

## 🚦 Next Steps for Karthik

### 1. Download ISIC Dataset

```bash
# Visit https://challenge.isic-archive.com/data/
# Download ISIC 2018 or 2020 dataset
# Extract to a folder, e.g., ~/Downloads/ISIC_2018/

# Preprocess
python scripts/preprocess_isic.py \
    --input_dir ~/Downloads/ISIC_2018 \
    --output_dir data
```

### 2. Train on Your NVIDIA GPU

```bash
# Start training (optimized for RTX GPUs with CUDA 11.8)
python -m src.train \
    --data_dir data/ham10000 \
    --epochs 30 \
    --batch_size 32

# Monitor training with:
# - Watch the console for real-time metrics
# - Check results.png for training curves
# - View metrics/confusion_matrix.png after training
```

**GPU Recommendations:**
- RTX 4060/4070 (8GB+): Batch size 32-64
- RTX 4050 (6GB): Batch size 16-24
- Mixed precision: Enabled by default (AMP)
- Expected training time: 45-60 minutes for 30 epochs on HAM10000

### 3. Deploy to Hugging Face Spaces

```bash
# After training:
# 1. Create account on huggingface.co
# 2. Create new Space (Streamlit SDK)
# 3. Upload files: app.py, requirements.txt, src/, models/model.pt
# 4. Space will auto-deploy!
```

### 4. Add to Resume

Use the bullets from `resume_bullet.txt` and update with your actual metrics!

---

**Ready to build? Start with:** `python scripts/create_dummy_data.py --split` to test the pipeline!
