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
BATCH_SIZE = 8
IMG_SIZE = 512

CATEGORIES = {
    "closures": 0,
    "upperbody": 1,
    "head": 2,
    "legs and feet": 3,
    "garment parts": 4,
    "arms and hands": 5,
    "decorations": 6,
    "neck": 7,
    "others": 8,
    "lowerbody": 9,
    "waist": 10,
    "wholebody": 11,
}
ENCODER_NAME = "efficientnet-b1"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = len(CATEGORIES)

torch.random.manual_seed(42)
numpy.random.seed(42)
