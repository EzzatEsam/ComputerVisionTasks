from pathlib import Path

import numpy
import torch


TRAIN_DATA_URL = "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip"
VAL_DATA_URL = "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip"
TRAIN_DATA_ANNOTATIONS = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json"
VAL_DATA_ANNOTATIONS = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json"
DATA_PATH = Path("data")

TRAIN_IMGS_FOLDER = DATA_PATH / "fashionist" / "train"
TRAIN_MASKS_FOLDER = DATA_PATH / "fashionist" / "train_masks"
VAL_IMGS_FOLDER = DATA_PATH / "fashionist" / "test"
VAL_MASKS_FOLDER = DATA_PATH / "fashionist" / "val_masks"
BATCH_SIZE = 32
IMG_SIZE = 512

CATEGORIES = {
    "background": 0,
    "closures": 1,
    "upperbody": 2,
    "head": 3,
    "legs and feet": 4,
    "garment parts": 5,
    "arms and hands": 6,
    "decorations": 7,
    "neck": 8,
    "others": 9,
    "lowerbody": 10,
    "waist": 11,
    "wholebody": 12,
}
ENCODER_NAME = "mit_b1"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = len(CATEGORIES)

torch.random.manual_seed(42)
numpy.random.seed(42)

# Normalization parameters for ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BEST_MODEL_PATH = Path.cwd() / "checkpoints" / "cloth-segmentation-epoch=07-val_loss=0.2172-val_iou=0.5409.ckpt"