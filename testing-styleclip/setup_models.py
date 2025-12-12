import bz2
import os
import subprocess
import sys
import urllib.request

# Import tqdm for progress bar
from tqdm import tqdm

from config import PRETRAINED_MODELS_PATH, TORCH_CUDA_ARCH_LIST

os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST

# Setup paths
save_path = PRETRAINED_MODELS_PATH
os.makedirs(save_path, exist_ok=True)

# Define URLs
e4e_download_url = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/encoder4editing/e4e_ffhq_encode.pt"
landmarks_model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
style_gan2_ada_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"

# Dictionary to manage download targets
download_targets = {
    "e4e_ffhq_encode.pt": e4e_download_url,
    "shape_predictor_68_face_landmarks.dat.bz2": landmarks_model_url,
    "ffhq.pkl": style_gan2_ada_url
}

class DownloadProgressBar(tqdm):
    """Provides a progress bar for URL downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    filepath = os.path.join(save_path, filename)
    print(f"Downloading {filename}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
        print(f"Successfully downloaded to {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to download {filename} from {url}. Error: {e}")
        return None

def extract_bz2(filepath):
    """Extracts a .bz2 compressed file."""
    print(f"Extracting {filepath}...")
    new_filepath = filepath.rstrip('.bz2')
    try:
        with bz2.open(filepath, 'rb') as f_in:
            with open(new_filepath, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(filepath)  # Remove the compressed file after extraction
        print(f"Extracted to {new_filepath}")
        return new_filepath
    except Exception as e:
        print(f"Failed to extract {filepath}. Error: {e}")
        return None


if __name__ == "__main__":
    # --- Main Download Logic ---
    print("Starting download of pre-trained models...")

    # 1. Download the e4e encoder
    e4e_path = download_file(download_targets["e4e_ffhq_encode.pt"], "e4e_ffhq_encode.pt")

    # 2. Download and extract the facial landmarks model
    bz2_path = download_file(download_targets["shape_predictor_68_face_landmarks.dat.bz2"], "shape_predictor_68_face_landmarks.dat.bz2")
    if bz2_path:
        landmarks_path = extract_bz2(bz2_path)

    # 3. Try to download the StyleGAN2 FFHQ model (URL may not work)
    ffhq_path = download_file(download_targets["ffhq.pkl"], "ffhq.pkl")

    print("\nDownload process complete.")
    print("-" * 50)


    # 4. Navigate to lib/style_clip_pytorch and run extract.py
    print("Extracting StyleCLIP text/image directions...")
    style_clip_path = os.path.join(os.getcwd(), "lib", "style_clip_pytorch")
    if os.path.exists(style_clip_path):
        original_cwd = os.getcwd()
        os.chdir(style_clip_path)
        try:
            subprocess.run([sys.executable, "extract.py"], check=True)
            print("Extraction completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run extract.py. Error: {e}")
        finally:
            os.chdir(original_cwd)

    # Post-download summary and alternative steps
    if ffhq_path is None:
        print("IMPORTANT: The FFHQ StyleGAN2 model failed to download.")
        print("This is a common issue as the NVIDIA URL often changes[citation:5].")
        print("\nYou can try the following:")
        print("1. Check the official StyleGAN2-ADA-PyTorch GitHub repo for updated links.")
        print("2. Search for 'ffhq.pkl' on platforms like Kaggle Datasets.")
        print("3. Use a different pre-trained StyleGAN2 model compatible with StyleCLIP.")
    else:
        print("All files downloaded successfully.")

    print(f"\nYou can find the models in: {os.path.abspath(save_path)}")


