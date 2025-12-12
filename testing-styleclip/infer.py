import argparse
import os
import sys
import warnings
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image

from config import PRETRAINED_MODELS_PATH, TORCH_CUDA_ARCH_LIST

# Path Setup
ROOT = Path(__file__).resolve().parent
LIB_PATH = ROOT / "lib" / "style_clip_pytorch"
sys.path.insert(0, str(LIB_PATH))

# Local Imports
from lib.style_clip_pytorch.embedding import get_delta_t  # noqa: E402
from lib.style_clip_pytorch.manipulator import Manipulator  # noqa: E402
from lib.style_clip_pytorch.mapper import get_delta_s  # noqa: E402
from lib.style_clip_pytorch.wrapper import Generator  # noqa: E402 

warnings.filterwarnings("ignore")
os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST


def save_result(tensor : torch.Tensor, output_path : str):
    img = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    Image.fromarray(img[0].cpu().numpy()).save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input image path")
    parser.add_argument("-o", "--output", type=str, default="output.png", help="Output path")
    parser.add_argument("-t", "--target", type=str, required=True, help="Target text prompt")
    parser.add_argument("-n", "--neutral", type=str, default="face", help="Neutral text prompt")
    parser.add_argument("-a", "--alpha", type=float, default=2, help="Manipulation strength")
    parser.add_argument("-b", "--beta", type=float, default=0.1, help="Disentanglement threshold")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    # Load Models
    print(f"Loading models on {args.device}...")
    ckpt = str(Path(PRETRAINED_MODELS_PATH) / "ffhq.pkl")
    G = Generator(ckpt, args.device)
    model, _ = clip.load("ViT-B/32", device=args.device)
    fs3 = np.load(LIB_PATH / "tensor" / "fs3.npy")

    manipulator = Manipulator(
        G, args.device,
        detector_model_path=Path(PRETRAINED_MODELS_PATH) / "shape_predictor_68_face_landmarks.dat",
        tensors_parent_path=LIB_PATH / "tensor",
    )

    # Inversion
    print(f"Processing {args.input}...")
    manipulator.set_real_img_projection(args.input, inv_mode='w+', pti_mode='s')

    # Calculate Directions
    classnames = [ args.target,args.neutral]
    delta_t = get_delta_t(classnames, model)
    delta_s, num_channels = get_delta_s(fs3, delta_t, manipulator, beta_threshold=args.beta)
    print(f"Num channels: {num_channels}")
    print(f"Manipulating: '{args.neutral}' -> '{args.target}' (Alpha: {args.alpha}, Beta: {args.beta})")

    # Synthesis
    manipulator.set_alpha([args.alpha])
    styles = manipulator.manipulate(delta_s)
    result_tensor = manipulator.synthesis_from_styles(styles, 0, 1)[0]

    save_result(result_tensor, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()