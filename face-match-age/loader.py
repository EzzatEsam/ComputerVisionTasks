import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from config import (
    BATCH_SIZE,
    IMAGES_FEATURES_FILE,
    NUM_WORKERS,
    TRAIN_IMGS_FOLDER,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
    VAL_IMGS_FOLDER,
    IMG_SIZE,
)

df = pd.read_csv(IMAGES_FEATURES_FILE)
NAME_TO_AGE = dict(zip(df["name"], df["age"]))
del df


class CacdDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        """
        Args:
            root_dir (Path): Path to either TRAIN_IMGS_FOLDER or VAL_IMGS_FOLDER
            transform (callable, optional): Transform to apply to the image
        """
        self.root_dir = root_dir
        self.transform = transform

        # 1. Load the CSV once to build a lookup dictionary
        # We assume columns "name" (filename) and "age" exist.

        # 2. List ONLY the images present in this specific directory (Train or Val)
        # We sort them to ensure deterministic ordering every time we load.
        self.img_paths = sorted(list(self.root_dir.glob("*.jpg")))

        # 3. Optional: Filter out images found in folder but not in CSV (safety check)
        # This ensures __getitem__ never fails on a missing key.
        self.valid_data = []
        for img_path in self.img_paths:
            img_name = img_path.name
            if img_name in NAME_TO_AGE:
                self.valid_data.append((img_path, NAME_TO_AGE[img_name]))
            else:
                print(f"Warning: Image {img_name} found in folder but not in CSV.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_path, age = self.valid_data[idx]

        # Read image
        image = read_image(str(img_path), mode=ImageReadMode.RGB)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return image and float age
        # (If using KL-Divergence later, you might convert this float to a distribution here)
        return image, torch.tensor(age, dtype=torch.float32)


# --- Transform Logic (Same as before) ---


def get_tv_transforms(train=True):
    transforms_list = []

    if train:
        transforms_list.append(
            v2.RandomResizedCrop(
                (IMG_SIZE, IMG_SIZE),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                antialias=True,
            )
        )
        transforms_list.append(v2.RandomHorizontalFlip(p=0.5))
        transforms_list.append(v2.RandomRotation(degrees=10))
        transforms_list.append(
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        )
    else:
        transforms_list.append(v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True))
        transforms_list.append(v2.CenterCrop(IMG_SIZE))

    transforms_list.append(v2.ToDtype(torch.float32, scale=True))
    transforms_list.append(v2.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD))

    return v2.Compose(transforms_list)


# --- Initialization ---

# 1. Train Dataset
train_ds = CacdDataset(
    root_dir=TRAIN_IMGS_FOLDER,
    transform=get_tv_transforms(train=True),
)

# 2. Validation Dataset
val_ds = CacdDataset(
    root_dir=VAL_IMGS_FOLDER,
    transform=get_tv_transforms(train=False),
)

# 3. DataLoaders
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2,
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import shutil

    # 1. Setup Output Directory
    save_dir = Path("visualized_samples")
    if save_dir.exists():
        shutil.rmtree(save_dir)  # Clean previous runs
    save_dir.mkdir(parents=True, exist_ok=True)

    def denormalize(tensor, mean, std):
        """
        Reverses the v2.Normalize operation for visualization.
        Input: Tensor (C, H, W)
        Output: Numpy Array (H, W, C) range [0, 1]
        """
        # Clone so we don't modify the data in the loader
        t = tensor.clone()

        # Reverse Normalization: pixel = (pixel * std) + mean
        # We loop over the 3 channels
        for t_c, m, s in zip(t, mean, std):
            t_c.mul_(s).add_(m)

        # Clamp values to [0, 1] to handle any float precision overshoots
        t = torch.clamp(t, 0, 1)

        # Permute from (C, H, W) -> (H, W, C) for Matplotlib
        return t.permute(1, 2, 0).numpy()

    def save_samples(loader, split_name):
        print(f"Fetching batch from {split_name} loader...")

        # Get one batch
        images, ages = next(iter(loader))

        # Pick 4 random indices from the batch
        indices = np.random.choice(len(images), 4, replace=False)

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle(f"{split_name.capitalize()} Samples", fontsize=20)

        for i, idx in enumerate(indices):
            img_tensor = images[idx]
            age = ages[idx].item()

            # Convert tensor to displayable image
            vis_img = denormalize(img_tensor, TRANSFORM_MEAN, TRANSFORM_STD)

            axes[i].imshow(vis_img)
            axes[i].set_title(f"Age: {age:.1f}", fontsize=14, color="blue")
            axes[i].axis("off")

        out_path = save_dir / f"{split_name}_samples.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

    # 2. Execute
    save_samples(train_loader, "train")
    save_samples(val_loader, "validation")

    print("\nVisualization complete. Check the 'visualized_samples' folder.")
