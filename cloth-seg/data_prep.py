import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import os
from pycocotools.coco import COCO
import numpy as np
import cv2
from config import (
    CATEGORIES,
    DATA_PATH,
    TRAIN_DATA_ANNOTATIONS,
    TRAIN_DATA_URL,
    VAL_DATA_ANNOTATIONS,
    VAL_DATA_URL,
)



def download_file(url: str, dest_path: Path, chunk_size: int = 8192):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"File already exists at {dest_path}")
        return

    print(f"Downloading {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with (
            open(dest_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    print("Download completed.")


def extract_zip(zip_path: Path, extract_dir: Path):
    print(f"Extracting {zip_path} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction completed.")


def generate_masks(json_file_path: Path, folder: Path):
    print(f"Generating masks from {json_file_path} into {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    coco = COCO(json_file_path)
    for img in tqdm(coco.getImgIds()):
        annotations = coco.getAnnIds(img)
        img_info = coco.loadImgs(img)[0]
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
        for ann_id in annotations:
            annotation = coco.loadAnns(ann_id)[0]
            mask_for_ann = coco.annToMask(annotation)
            cat_id = annotation["category_id"]
            super_cat= coco.loadCats(cat_id)[0]["supercategory"]
            mask[mask_for_ann == 1] = CATEGORIES[super_cat]
            

        output_filename = os.path.splitext(img_info["file_name"])[0] + ".png"
        save_path = folder / output_filename

        # cv2 writes images to disk
        cv2.imwrite(str(save_path), mask)
    print("Mask generation completed.")


def load_data():
    # download_file(TRAIN_DATA_URL, DATA_PATH / "fashionist.zip")
    # extract_zip(DATA_PATH / "fashionist.zip", DATA_PATH / "fashionist")
    # download_file(VAL_DATA_URL, DATA_PATH / "fashionist_val.zip")
    # extract_zip(DATA_PATH / "fashionist_val.zip", DATA_PATH / "fashionist")
    # download_file(
    #     TRAIN_DATA_ANNOTATIONS,
    #     DATA_PATH / "fashionist" / "instances_attributes_train2020.json",
    # )
    # download_file(
    #     VAL_DATA_ANNOTATIONS,
    #     DATA_PATH / "fashionist" / "instances_attributes_val2020.json",
    # )

    generate_masks(
        DATA_PATH / "fashionist" / "instances_attributes_train2020.json",
        DATA_PATH / "fashionist" / "train_masks",
    )
    generate_masks(
        DATA_PATH / "fashionist" / "instances_attributes_val2020.json",
        DATA_PATH / "fashionist" / "val_masks",
    )


if __name__ == "__main__":
    load_data()
