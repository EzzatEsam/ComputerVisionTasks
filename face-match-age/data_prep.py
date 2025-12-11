import requests
import tarfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import scipy.io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
from config import (
    CACHE_DIR,
    DATASET_FILENAME,
    DATASET_URL,
    TRAIN_IMGS_FOLDER,
    VAL_IMGS_FOLDER,
    VAL_RATIO,
    IMG_SIZE,
)


RESIZE_DIM = IMG_SIZE

# --- Helper Functions ---


def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        print(f"{dest_path.name} already exists. Skipping download.")
        return

    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with (
        open(dest_path, "wb") as file,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


def extract_tar(tar_path: Path, dest_folder: Path):
    # Expect 'imdb_crop' folder inside
    expected_folder_name = "imdb_crop"
    final_extract_path = dest_folder / expected_folder_name

    if final_extract_path.exists():
        print(
            f"Dataset already extracted to {final_extract_path}. Skipping extraction."
        )
        return final_extract_path

    print(f"Extracting {tar_path.name} (This takes a while)...")
    with tarfile.open(tar_path) as tar:
        # Fast extraction without progress bar for speed on huge archives
        tar.extractall(path=dest_folder)

    return final_extract_path


def mat_date_to_python(matlab_datenum: float) -> datetime:
    """Convert Matlab datenum to Python datetime."""
    try:
        return datetime.fromordinal(int(matlab_datenum)) - timedelta(days=366)
    except Exception:
        return None


def process_metadata(mat_path: Path) -> pd.DataFrame:
    """
    Vectorized processing of the IMDB metadata.
    Follows the standard cleanup: keep (face_score != -inf) and (second_face_score == NaN).
    """
    print(f"Loading metadata from {mat_path}...")
    mat = scipy.io.loadmat(mat_path)

    # Handle IMDB vs Wiki structure
    root_key = "imdb" if "imdb" in mat else "wiki"
    data = mat[root_key][0, 0]

    # Extract raw arrays (flattened)
    print("Extracting arrays...")
    dobs = data["dob"][0]
    photo_taken = data["photo_taken"][0]
    full_path = data["full_path"][0]
    gender = data["gender"][0]
    face_score = data["face_score"][0]
    second_face_score = data["second_face_score"][0]

    print("Applying filters...")

    # 1. Must have a face detected (Score is not -inf)
    mask_face_found = face_score > -np.inf

    # 2. Must NOT have a second face (Second score must be NaN)
    mask_single_face = np.isnan(second_face_score)

    # 3. Must have a valid gender (Not NaN)
    mask_gender_valid = ~np.isnan(gender)

    # Combine masks
    valid_mask = mask_face_found & mask_single_face & mask_gender_valid

    print(f"Total raw entries: {len(face_score)}")
    print(f"Entries passing face score/gender checks: {np.sum(valid_mask)}")

    # Apply mask to reduce data size BEFORE processing loop
    dobs = dobs[valid_mask]
    photo_taken = photo_taken[valid_mask]
    full_path = full_path[valid_mask]
    gender = gender[valid_mask]

    # --- CALCULATE AGE ---
    print("Calculating ages...")
    cleaned_data = []

    for i in tqdm(range(len(dobs))):
        dob_num = dobs[i]
        taken_year = photo_taken[i]

        birth_date = mat_date_to_python(dob_num)

        if birth_date is None:
            continue

        age = taken_year - birth_date.year

        # Sanity check: 0 to 100
        if 0 <= age <= 100:
            # Clean filename
            fpath = full_path[i][0]
            fname = os.path.basename(fpath)

            cleaned_data.append(
                {
                    "filename": fname,
                    "age": int(age),
                    "gender": int(gender[i]),
                    "original_path": fpath,
                }
            )

    df = pd.DataFrame(cleaned_data)
    print(f"Final valid images: {len(df)}")
    return df


def process_and_save(args: tuple[Path, Path]):
    """Resizes image and saves it."""
    src_path, dest_folder = args
    dest_path = dest_folder / src_path.name

    if dest_path.exists():
        return

    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize((RESIZE_DIM, RESIZE_DIM), Image.Resampling.BILINEAR)
            img.save(dest_path, "JPEG", quality=90)
    except Exception:
        print(f"Failed to process image: {src_path}")
        pass


# --- Main Execution ---

if __name__ == "__main__":
    # 0. Setup Cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = CACHE_DIR / DATASET_FILENAME

    # 1. Download
    download_file(DATASET_URL, tar_path)

    # 2. Extract
    raw_data_path = extract_tar(tar_path, CACHE_DIR)

    # 3. Process Metadata
    mat_file = (
        raw_data_path / "imdb.mat"
    )  # It is usually named imdb.mat inside imdb_crop

    if not mat_file.exists():
        # Fallback check if it's strictly wiki
        if (raw_data_path / "wiki.mat").exists():
            mat_file = raw_data_path / "wiki.mat"
        else:
            raise FileNotFoundError(
                f"Could not find .mat metadata file in {raw_data_path}"
            )

    df_meta = process_metadata(mat_file)

    # 4. Locate Images on Disk
    print("Mapping metadata to files on disk...")
    found_files = {}
    valid_filenames = set(df_meta["filename"].values)

    # Scan directory
    # IMDB structure is nested: imdb_crop/00/file.jpg, etc.
    for f in tqdm(raw_data_path.rglob("*.jpg"), desc="Scanning disk"):
        if f.name in valid_filenames:
            found_files[f.name] = f

    # Filter DataFrame to only include files we actually found
    df_final = df_meta[df_meta["filename"].isin(found_files.keys())].copy()
    print(f"Images matched with metadata: {len(df_final)}")

    if len(df_final) == 0:
        raise ValueError("No valid images found after filtering!")

    # 5. Shuffle and Split
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(VAL_RATIO * len(df_final))
    val_df = df_final.iloc[:split_idx]
    train_df = df_final.iloc[split_idx:]

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # 6. Create Output Directories
    TRAIN_IMGS_FOLDER.mkdir(parents=True, exist_ok=True)
    VAL_IMGS_FOLDER.mkdir(parents=True, exist_ok=True)

    # 8. Prepare Jobs
    train_jobs = [
        (found_files[row.filename], TRAIN_IMGS_FOLDER) for row in train_df.itertuples()
    ]
    val_jobs = [
        (found_files[row.filename], VAL_IMGS_FOLDER) for row in val_df.itertuples()
    ]
    all_jobs = train_jobs + val_jobs

    # 9. Execute Image Resize
    print(f"Resizing and saving images to {RESIZE_DIM}x{RESIZE_DIM}...")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_and_save, all_jobs), total=len(all_jobs)))

    #
    files_in_train = list(TRAIN_IMGS_FOLDER.glob("*.jpg"))
    files_in_val = list(VAL_IMGS_FOLDER.glob("*.jpg"))

    expected_train_count = len(train_df)
    expected_val_count = len(val_df)

    print(f"Expected images in train folder: {expected_train_count}")
    print(f"Expected images in val folder: {expected_val_count}")
    print(f"Total images in train folder: {len(files_in_train)}")
    print(f"Total images in val folder: {len(files_in_val)}")

    train_df = train_df[
        train_df["filename"].isin([f.name for f in files_in_train])
    ].reset_index(drop=True)
    val_df = val_df[
        val_df["filename"].isin([f.name for f in files_in_val])
    ].reset_index(drop=True)
    # 11. Save CSVs
    base_output_dir = TRAIN_IMGS_FOLDER.parent
    train_df.to_csv(base_output_dir / "train.csv", index=False)
    val_df.to_csv(base_output_dir / "val.csv", index=False)
    print(f"Saved train.csv and val.csv to {base_output_dir}")

    # 12. Cleanup Cache
    print("Cleaning up cache...")
    shutil.rmtree(CACHE_DIR)

    print("Dataset preparation completed.")
