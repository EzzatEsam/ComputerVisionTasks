# Face Matching and Age Prediction

A deep learning system for predicting human age from facial images and verifying if two face images belong to the same person, even across significant age differences.

## Features

- **Age Prediction:** Estimates age (0-100 years).
- **Face Verification:** Identifies if two face images are the same person using cosine similarity
- **Visual Output:** Generates annotated comparison images with predictions and match results
- **GPU Acceleration:** Supports CUDA/CPU.

## Quick Start

### Prerequisites

- Python 3.13+ (recommended)
- CUDA-capable GPU (optional)
- ~15GB disk space for dataset and models

## Prerequisites

This project is managed using **uv**. Ensure you have it installed:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd face-match-age
```

2. **Install dependencies:**
```bash
uv sync
```

### Dataset Setup

The system uses the IMDB filtered version.

**Automatic Download**
```bash
python data_prep.py
```

This will:
- Download the dataset
- Resize images to 160×160 pixels
- Split into train/validation sets (80/20)
- Save to `data/IMDB/train_images/` and `data/IMDB/val_images/`


## Usage

### Training

Train the age estimation model from scratch:

```bash
python train.py
```

**Resume from checkpoint:**
```bash
python train.py --ckpt checkpoints/last.ckpt
```

**Training Configuration:**
- Batch size: 256 (adjust in `config.py` if GPU memory limited)
- Max epochs: 50
- Early stopping: Patience of 6 epochs
- Learning rate: 1e-4 (Adam optimizer)
- Mixed precision: FP16 (automatic on GPU)

**Monitor Training with TensorBoard:**
```bash
tensorboard --logdir logs/face_age_det
```

Visit `http://localhost:6006` to view:
- Training/validation loss curves
- MAE metrics over time
- Sample predictions with visualizations

### Inference

Compare two face images and predict ages:

```bash
python infer.py path/to/image1.jpg path/to/image2.jpg output_folder
```

**Example:**
```bash
python infer.py \
    data/test/person_young.jpg \
    data/test/person_old.jpg \
    results/ \
    --json
```

**Arguments:**
- `img1`: Path to first image
- `img2`: Path to second image
- `output_folder`: Directory to save results
- `--json`: (Optional) Save detailed JSON output
- `--ckpt`: (Optional) Path to custom checkpoint (default: uses best model)

**Output Files:**
- `{name1}_vs_{name2}_result.jpg`: Annotated comparison image
- `{name1}_vs_{name2}_data.json`: Detailed JSON data (if `--json` flag used)

**Example JSON Output:**
```json
{
    "image_1": "person_young.jpg",
    "image_2": "person_old.jpg",
    "face_1": {
        "age_prediction": 28.45,
        "bbox": [120, 80, 340, 300]
    },
    "face_2": {
        "age_prediction": 67.23,
        "bbox": [150, 100, 370, 320]
    },
    "verification": {
        "is_match": true,
        "similarity_score": 0.8234,
        "threshold_used": 0.7
    }
}
```

### Configuration

Edit `config.py` to customize:

```python
# Dataset paths
DOWNLOAD_PATH = Path("./data/cacd")
TRAIN_IMGS_FOLDER = DOWNLOAD_PATH / "train_images"
VAL_IMGS_FOLDER = DOWNLOAD_PATH / "val_images"

# Training parameters
BATCH_SIZE = 256  # Reduce to 128 or 64 if GPU memory limited
IMG_SIZE = 160    # Model input size (don't change without retraining)
NUM_WORKERS = 8   # Adjust based on CPU cores

# Inference parameters
BEST_MODEL_PATH = Path("./checkpoints/cloth-segmentation-epoch=21-val_loss=0.8723-val_mae=3.57.ckpt")
VERIFICATION_THRESHOLD = 0.7  # Cosine similarity threshold (0.0-1.0)
```

## Project Structure

```
face-match-age/
├── config.py              # Configuration parameters
├── data_prep.py           # Dataset download and preprocessing
├── loader.py              # PyTorch Dataset and DataLoader definitions
├── model.py               # Age estimation model architecture
├── train.py               # Training script
├── infer.py               # Inference script
├── callbacks.py           # Training callbacks (image logging)
├── pyproject.toml         # Python dependencies
├── README.md              # This file
├── REPORT.md              # Detailed technical report
├── notebook.ipynb         # Jupyter notebook for experimentation
├── data/
│   └── IMDB/
│       ├── train_images/  # Training images (80%)
│       ├── val_images/    # Validation images (20%)
├── checkpoints/           # Saved model checkpoints
│   ├── last.ckpt          # Latest checkpoint
│   └── best.ckpt          # Best checkpoint
├── logs/                  # TensorBoard logs
└── visualized_samples/    # Example inference results
```

## Model Architecture

### Age Prediction Model
- **Backbone:** InceptionResnetV1 pretrained on VGGFace2
- **Head:** Custom regression head (512 → 256 → 101 age bins)
- **Loss:** KL Divergence with Gaussian target distributions
- **Output:** Age probability distribution (0-100 years)

### Face Verification Model
- **Model:** InceptionResnetV1 pretrained on VGGFace2 (frozen)
- **Method:** Cosine similarity of 512-D embeddings
- **Threshold:** 0.7 (configurable)
- **Output:** Binary match decision + similarity score

See [REPORT.md](REPORT.md) for detailed architecture explanation and performance analysis.


## Advanced Usage

### Custom Dataset

To train on your own dataset:

1. **Prepare images:**
   - Organize into train/val folders
   - Create CSV with columns: `name` (filename), `age` (integer)

2. **Update `config.py`:**
   ```python
   TRAIN_IMGS_FOLDER = Path("path/to/train")
   VAL_IMGS_FOLDER = Path("path/to/val")
   TRAIN_CSV_PATH = PATH_TO_TRAIN_CSV
   VAL_CSV_PATH = PATH_TO_VAL_CSV
   ```

3. **Train:**
   ```bash
   python train.py
   ```


### API Integration

```python
import torch
from model import AgeEstimationModel
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(device=device)
age_model = AgeEstimationModel.load_from_checkpoint("checkpoints/best.ckpt").to(device).eval()
verif_model = InceptionResnetV1(pretrained="vggface2").to(device).eval()

# Inference function
def predict_age_and_match(img1_path, img2_path):
    # Load images
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    
    # Detect faces
    face1 = mtcnn(img1).unsqueeze(0).to(device)
    face2 = mtcnn(img2).unsqueeze(0).to(device)
    
    # Predict ages
    with torch.no_grad():
        age1 = age_model(face1).softmax(1) @ torch.arange(101.0).to(device)
        age2 = age_model(face2).softmax(1) @ torch.arange(101.0).to(device)
        
        # Compute similarity
        emb1 = verif_model(face1)
        emb2 = verif_model(face2)
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    
    return {
        "age1": age1.item(),
        "age2": age2.item(),
        "similarity": similarity,
        "is_match": similarity > 0.7
    }
```
