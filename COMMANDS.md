# Quick Command Reference

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

```bash
# Generate dummy data for testing
python scripts/create_dummy_data.py --split --num_per_class 30

# Preprocess ISIC dataset
python scripts/preprocess_isic.py \
    --input_dir path/to/ISIC_raw \
    --output_dir data \
    --metadata metadata.csv
```

## Training

```bash
# Basic training
python -m src.train --data_dir data --epochs 20

# Training with custom settings
python -m src.train \
    --data_dir data \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-4

# Training on dummy data (quick test)
python -m src.train --data_dir data/dummy --epochs 5 --batch_size 8
```

## Inference

```bash
# Single image prediction
python -c "
from src.inference import predict_from_path
result = predict_from_path('image.jpg')
print(f'Prediction: {result[\"label\"]} ({result[\"confidence\"]:.2%})')
"

# Batch prediction
python -c "
from src.inference import predict_batch
results = predict_batch(['img1.jpg', 'img2.jpg'])
for r in results:
    print(f'{r[\"image_path\"]}: {r[\"label\"]}')
"
```

## Demo App

```bash
# Run Streamlit app
streamlit run app.py

# Run on different port
streamlit run app.py --server.port 8502
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_dataset.py -v
```

## Git Commands

```bash
# View commit history
git log --oneline

# Check status
git status

# Add remote (if pushing to GitHub)
git remote add origin https://github.com/yourusername/ai-skin-lesion-xai.git
git branch -M main
git push -u origin main
```

## Docker (Optional)

```bash
# Build image
docker build -t skin-lesion-classifier .

# Run container
docker run -p 8501:8501 skin-lesion-classifier
```

## Helpful Python Commands

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check model parameters
python -c "
from src.model_builder import build_model, count_parameters
model = build_model()
print(f'Parameters: {count_parameters(model):,}')
"

# View config
python -c "from src.config import print_config; print_config()"

# Test transforms
python -c "
from src.dataset import get_train_transforms
import numpy as np
transform = get_train_transforms()
img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
result = transform(image=img)
print(f'Output shape: {result[\"image\"].shape}')
"
```

## Monitoring Training

```bash
# Watch training in real-time (PowerShell)
Get-Content runs/training_log.txt -Wait

# View results
start results.png  # Windows
open results.png   # Mac
xdg-open results.png  # Linux

# Check last run metrics
cat runs/last_run.json
```

## Quick Smoke Test

```bash
# Full pipeline test (5 minutes)
python scripts/create_dummy_data.py --split --num_per_class 20
python -m src.train --data_dir data/dummy --epochs 3 --batch_size 8
streamlit run app.py
```

## Troubleshooting

```bash
# CUDA out of memory
python -m src.train --data_dir data --batch_size 8

# Module not found
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.10+

# Clear GPU cache (in Python)
python -c "import torch; torch.cuda.empty_cache()"

# Re-install dependencies
pip install --force-reinstall -r requirements.txt
```

## Performance Benchmarking

```bash
# Time single inference
python -c "
import time
from src.inference import predict_from_path
import numpy as np
from PIL import Image

# Create test image
Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save('test.jpg')

start = time.time()
result = predict_from_path('test.jpg')
print(f'Inference time: {time.time() - start:.3f}s')
"

# Profile training
python -m cProfile -o profile.stats -m src.train --data_dir data/dummy --epochs 1
```
