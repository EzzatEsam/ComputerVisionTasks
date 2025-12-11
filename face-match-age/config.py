import os
from pathlib import Path


DOWNLOAD_PATH = Path("./data")
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

# IMAGES_PARENT_PATH = Path("./data/cacd/cacd_split/IMDB")
# DATASET_NAME = "pdombrza/cacd-filtered-dataset"
TRAIN_IMGS_FOLDER = DOWNLOAD_PATH / "train_images"
VAL_IMGS_FOLDER = DOWNLOAD_PATH / "val_images"
IMG_SIZE = 160
NUM_WORKERS = os.cpu_count() - 2 if os.cpu_count() > 2 else 1

VAL_RATIO = 0.2
BATCH_SIZE = 256

TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]


BEST_MODEL_PATH=Path("./checkpoints/face-age-detepoch=10-val_loss=1.2647-val_mae=5.54.ckpt")
VERIFICATION_THRESHOLD = 0.7  # Cosine similarity threshold