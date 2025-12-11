import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_IMGS_FOLDER,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
    VAL_IMGS_FOLDER,
    IMG_SIZE,
)

class ImdbWikiDataset(Dataset):
    def __init__(self, root_dir: Path, csv_file: Path, transform=None):
        """
        Args:
            root_dir (Path): Path to the folder containing images.
            csv_file (Path): Path to the specific train.csv or val.csv.
            transform (callable, optional): Transform to apply to the image.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load the specific CSV for this split
        if not csv_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_file}")
            
        self.df = pd.read_csv(csv_file)
        
        # We expect columns: 'filename', 'age', 'gender'
        # Check required columns exist
        required_cols = ['filename', 'age']
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row data
        row = self.df.iloc[idx]
        filename = row['filename']
        age = row['age']
        # gender = row['gender'] # Available if needed later

        # Construct full path
        img_path = self.root_dir / filename

        # Read image
        try:
            # We assume images are valid since we cleaned them in the prep script
            image = read_image(str(img_path), mode=ImageReadMode.RGB)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image or handle error appropriately in production
            image = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Return image and float age
        return image, torch.tensor(age, dtype=torch.float32)


# --- Transform Logic (Unchanged) ---

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

# Define paths to the CSV files generated in the previous step
# They are located in the parent folder of the image directories
TRAIN_CSV_PATH = TRAIN_IMGS_FOLDER.parent / "train.csv"
VAL_CSV_PATH = VAL_IMGS_FOLDER.parent / "val.csv"

# 1. Train Dataset
train_ds = ImdbWikiDataset(
    root_dir=TRAIN_IMGS_FOLDER,
    csv_file=TRAIN_CSV_PATH,
    transform=get_tv_transforms(train=True),
)

# 2. Validation Dataset
val_ds = ImdbWikiDataset(
    root_dir=VAL_IMGS_FOLDER,
    csv_file=VAL_CSV_PATH,
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

    # 1. Setup Output Directory
    save_dir = Path("visualized_samples")

    save_dir.mkdir(parents=True, exist_ok=True)

    def denormalize(tensor, mean, std):
        """Reverses v2.Normalize for visualization."""
        t = tensor.clone()
        for t_c, m, s in zip(t, mean, std):
            t_c.mul_(s).add_(m)
        t = torch.clamp(t, 0, 1)
        return t.permute(1, 2, 0).numpy()

    def save_samples(loader, split_name):
        print(f"Fetching batch from {split_name} loader...")

        try:
            images, ages = next(iter(loader))
        except StopIteration:
            print(f"Loader {split_name} is empty.")
            return

        # Pick 4 random indices from the batch
        batch_len = len(images)
        indices = np.random.choice(batch_len, min(4, batch_len), replace=False)

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle(f"{split_name.capitalize()} Samples", fontsize=20)

        if len(indices) < 4:
            # Handle edge case if batch size < 4
            axes = axes.flatten()

        for i, idx in enumerate(indices):
            img_tensor = images[idx]
            age = ages[idx].item()

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

    print(f"\nDataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}")
    print("Visualization complete. Check the 'visualized_samples' folder.")