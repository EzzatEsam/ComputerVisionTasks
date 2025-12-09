import kagglehub
from pathlib import Path
import random
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from config import (
    DATASET_NAME,
    TRAIN_IMGS_FOLDER,
    VAL_IMGS_FOLDER,
    VAL_RATIO,
    IMG_SIZE, # Ensure this is imported (should be 160 or 256)
)

# Define the target size slightly larger than model input if you want to do random cropping
# or exactly the model input size (160) for maximum speed.
# Facenet default is 160.
RESIZE_DIM = IMG_SIZE 

def process_and_save(args):
    """
    Reads image, resizes it, and saves it to the destination.
    Args: tuple (src_path, dest_folder)
    """
    src_path, dest_folder = args
    dest_path = dest_folder / src_path.name
    
    # 1. Skip if already exists (allows restarting script if it crashes)
    if dest_path.exists():
        return

    try:
        with Image.open(src_path) as img:
            # 2. Convert to RGB (handle PNG/RGBA or Grayscale issues)
            img = img.convert('RGB')
            
            # 3. Resize (Bilinear is fast and good enough for faces)
            img = img.resize((RESIZE_DIM, RESIZE_DIM), Image.Resampling.BILINEAR)
            
            # 4. Save as JPG with reasonable compression
            img.save(dest_path, "JPEG", quality=90)
            
    except Exception as e:
        print(f"Error processing {src_path.name}: {e}")

if __name__ == "__main__":
    # 1. Download Dataset
    print("Downloading dataset...")
    dl_path = kagglehub.dataset_download(DATASET_NAME)
    dl_path = Path(dl_path)
    print(f"Dataset downloaded to cache: {dl_path}")

    # Note: We do NOT copy the whole raw folder to DOWNLOAD_PATH anymore. 
    # We will read from cache and write processed images to DOWNLOAD_PATH.
    
    # 2. Locate Images in the Cache
    # We look inside the cached download path for the structure
    # Adjust the glob pattern if the Kaggle structure is nested
    print("Scanning for images...")
    # Heuristic: Try to find the inner folder if IMAGES_PARENT_PATH was relative
    # If IMAGES_PARENT_PATH in config was absolute/hardcoded, we might need to adjust logic.
    # Let's search recursively in the downloaded cache.
    all_img_paths = list(dl_path.rglob("*.jpg"))
    
    # Filter out duplicates if any
    img2path = {i.name: i for i in all_img_paths}
    img_names = list(img2path.keys())
    
    print(f"Total images found: {len(img_names)}")
    
    if len(img_names) == 0:
        raise ValueError("No images found! Check the kaggle dataset structure.")

    # 3. Shuffle and Split
    random.seed(42)
    random.shuffle(img_names)

    split_idx = int(VAL_RATIO * len(img_names))
    val_img_names = img_names[:split_idx]
    train_img_names = img_names[split_idx:]

    print(f"Training images: {len(train_img_names)}")
    print(f"Validation images: {len(val_img_names)}")

    # 4. Create Directories
    TRAIN_IMGS_FOLDER.mkdir(parents=True, exist_ok=True)
    VAL_IMGS_FOLDER.mkdir(parents=True, exist_ok=True)

    # 5. Prepare Job Lists
    # List of tuples: (path_to_source, folder_to_save)
    train_jobs = [(img2path[name], TRAIN_IMGS_FOLDER) for name in train_img_names]
    val_jobs = [(img2path[name], VAL_IMGS_FOLDER) for name in val_img_names]
    all_jobs = train_jobs + val_jobs

    # 6. Execute in Parallel (Uses all CPU Cores)
    # This resolves the "CPU Bottleneck" before training even starts.
    print(f"Resizing and saving images to {RESIZE_DIM}x{RESIZE_DIM}...")
    
    # max_workers=None defaults to number of processors on the machine
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_and_save, all_jobs), total=len(all_jobs)))

    print("Dataset preparation completed.")