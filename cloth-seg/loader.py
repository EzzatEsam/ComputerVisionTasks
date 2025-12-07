from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
from torchvision import tv_tensors 
import matplotlib.pyplot as plt
from config import (
    BATCH_SIZE,
    IMG_SIZE,
    TRAIN_IMGS_FOLDER,
    TRAIN_MASKS_FOLDER,
    VAL_IMGS_FOLDER,
    VAL_MASKS_FOLDER,
)
    

class FashionpediaDatasetTV(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, transforms=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms

        # Sort to ensure alignment
        self.image_ids = sorted([f.name for f in self.images_dir.iterdir()])
        self.mask_ids = sorted([f.name for f in self.masks_dir.iterdir()])

        # Basic validation
        if len(self.image_ids) != len(self.mask_ids):
            print(
                f"Warning: Mismatch in file counts. Imgs: {len(self.image_ids)}, Masks: {len(self.mask_ids)}"
            )
            # Filter to intersection (same logic as before)
            img_stems = set(f.split(".")[0] for f in self.image_ids)
            mask_stems = set(f.split(".")[0] for f in self.mask_ids)
            valid = sorted(list(img_stems.intersection(mask_stems)))
            self.image_ids = [f"{x}.jpg" for x in valid]
            self.mask_ids = [f"{x}.png" for x in valid]
            
        # Check empty file :
        for i,(img_name , mask_name) in enumerate(zip(self.image_ids, self.mask_ids)):
            img_path = self.images_dir / img_name
            mask_path = self.masks_dir / mask_name
            if img_path.stat().st_size == 0 or mask_path.stat().st_size == 0:
                print(f"Warning: Empty file detected. Img: {img_path}, Mask: {mask_path}")
                self.image_ids[i] = None
                self.mask_ids[i] = None
                
        # Remove None entries
        self.image_ids = [x for x in self.image_ids if x is not None]
        self.mask_ids = [x for x in self.mask_ids if x is not None]

                

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.images_dir / self.image_ids[idx]
        # read_image returns [3, H, W] uint8 Tensor
        img = read_image(str(img_path), mode=ImageReadMode.RGB)

        # 2. Load Mask
        mask_path = self.masks_dir / self.mask_ids[idx]
        # read_image returns [1, H, W] uint8 Tensor
        mask = read_image(str(mask_path), mode=ImageReadMode.GRAY)

        # 3. Wrap Mask in tv_tensors
        # This tells torchvision: "Don't normalize these values, but DO flip/rotate them"
        mask = tv_tensors.Mask(mask)

        # 4. Apply Transforms (V2 style)
        if self.transforms:
            img, mask = self.transforms(img, mask)

        # 5. Final Cleanup
        # Remove the channel dim from mask: [1, H, W] -> [H, W]
        mask = mask.squeeze(0)

        # Convert to Long (int64) for CrossEntropyLoss
        mask = mask.long()

        return img, mask


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_tv_transforms(train=True):
    transforms_list = []

    # 1. Resize/Crop (Applied to both Image and Mask)
    if train:
        # RandomResizedCrop replaces fixed Resize for training variance
        transforms_list.append(v2.RandomResizedCrop(
            (IMG_SIZE, IMG_SIZE), 
            scale=(0.8, 1.0), 
            ratio=(0.75, 1.33), 
            antialias=True
        ))
    else:
        # Fixed Resize and a Center Crop for validation/test consistency
        transforms_list.append(v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True))
        transforms_list.append(v2.CenterCrop(IMG_SIZE))

    if train:
        # 2. Random Geometric & Color Augmentations
        transforms_list.append(v2.RandomHorizontalFlip(p=0.5))
        transforms_list.append(v2.RandomRotation(degrees=10))
        
        # Add slight translation and shear
        transforms_list.append(v2.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            shear=10 
        ))

        # Color Augmentations (affecting Image only)
        transforms_list.append(
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        )
        transforms_list.append(v2.RandomGrayscale(p=0.1))

    # 3. Type Conversion & Normalization (Applied last)
    transforms_list.append(v2.ToDtype(torch.float32, scale=True))
    transforms_list.append(v2.Normalize(mean=MEAN, std=STD))

    return v2.Compose(transforms_list)


train_ds = FashionpediaDatasetTV(
    TRAIN_IMGS_FOLDER, TRAIN_MASKS_FOLDER, transforms=get_tv_transforms(train=True)
)

val_ds = FashionpediaDatasetTV(
    VAL_IMGS_FOLDER, VAL_MASKS_FOLDER, transforms=get_tv_transforms(train=False)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4 , pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 , pin_memory=True)





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    # -------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------
    SAVE_DIR = Path("visualized_samples")
    SAVE_DIR.mkdir(exist_ok=True)
    
    # How many images from the batch to display/save
    NUM_SAMPLES_TO_SHOW = 4 

    # ImageNet stats for denormalization (Must match your transforms)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    print("Fetching a batch of data...")
    # Get a single batch from the loader
    images, masks = next(iter(train_loader))
    
    print(f"Batch Image Shape: {images.shape}") # Expect [B, 3, H, W]
    print(f"Batch Mask Shape: {masks.shape}")   # Expect [B, H, W]

    # -------------------------------------------------
    # VISUALIZATION LOOP
    # -------------------------------------------------
    for i in range(min(NUM_SAMPLES_TO_SHOW, len(images))):
        img_tensor = images[i]
        mask_tensor = masks[i]

        # 1. Denormalize the Image for display
        # Formula: image = (tensor * std) + mean
        img_vis = img_tensor * STD + MEAN
        
        # Clamp values to [0, 1] to avoid matplotlib warnings
        img_vis = torch.clamp(img_vis, 0, 1)
        
        # Convert to Numpy and change shape from (C, H, W) -> (H, W, C)
        img_np = img_vis.permute(1, 2, 0).numpy()
        mask_np = mask_tensor.numpy()

        # 2. Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original Image
        ax[0].imshow(img_np)
        ax[0].set_title(f"Input Image (Sample {i})")
        ax[0].axis('off')
        
        # Ground Truth Mask
        # We use 'nipy_spectral' or 'tab20' to make different integer IDs look distinct
        # We use 'nearest' interpolation to keep edges sharp (pixelated) rather than blurry
        cmap = plt.get_cmap('nipy_spectral')
        
        # Determine unique classes in this specific mask for the title
        unique_classes = np.unique(mask_np)
        
        ax[1].imshow(mask_np, cmap=cmap, interpolation='nearest')
        ax[1].set_title(f"Target Mask (Classes: {unique_classes})")
        ax[1].axis('off')

        # 3. Save to disk
        save_path = SAVE_DIR / f"train_sample_{i}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved visualization to: {save_path}")
        
        # 4. Display on screen
        plt.show()

    print("Done.")
