import argparse
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import v2

# Import your project modules
from model import AgeEstimationModel
from config import (
    BEST_MODEL_PATH,
    IMG_SIZE,
    TRANSFORM_MEAN,
    TRANSFORM_STD,
    VERIFICATION_THRESHOLD,
)

# --- Configuration ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_age_transforms():
    """Transforms specifically for the Age Estimation Model (ImageNet stats)."""
    return v2.Compose(
        [
            v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=TRANSFORM_MEAN, std=TRANSFORM_STD),
        ]
    )


def load_models(ckpt_path: Path):
    print(f"Loading models on {DEVICE}...")

    # 1. Face Detector (MTCNN)
    mtcnn = MTCNN(
        image_size=160, margin=0, keep_all=False, post_process=True, device=DEVICE
    )

    # 2. Age Model (Fine-Tuned)
    age_model = AgeEstimationModel.load_from_checkpoint(ckpt_path)
    age_model.to(DEVICE).eval()
    age_model.freeze()

    # 3. Verification Model (Pretrained VGGFace2)
    verif_model = (
        InceptionResnetV1(pretrained="vggface2", classify=False).to(DEVICE).eval()
    )

    return mtcnn, age_model, verif_model


def process_single_image(
    img_path: Path,
    mtcnn: MTCNN,
    age_model: AgeEstimationModel,
    verif_model: InceptionResnetV1,
    age_transforms: v2.Compose,
):
    """
    Detects face, extracts embeddings, and predicts age for a single image.
    """
    try:
        # Load as PIL RGB
        img_pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

    # 1. Detect Face (Get Box)
    # boxes: [x1, y1, x2, y2]
    boxes, probs = mtcnn.detect(img_pil)

    if boxes is None or len(boxes) == 0:
        print(f"No face detected in {img_path}")
        return None

    # Pick the face with highest probability
    best_idx = np.argmax(probs)
    box = boxes[best_idx]  # [x1, y1, x2, y2]

    # 2. Prepare Input for Age Model (Custom Norm)
    # We crop using the box, then transform
    face_crop = img_pil.crop(box)
    age_input = age_transforms(face_crop).unsqueeze(0).to(DEVICE)

    # 3. Prepare Input for Verification Model (Standard Norm [-1, 1])
    verif_input = v2.Compose(
        [ 
            v2.ToImage(),
            v2.Resize((160, 160), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )(face_crop).unsqueeze(0).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        # Age Prediction (Expected Value)
        age_logits = age_model(age_input)
        age_probs = torch.softmax(age_logits, dim=1)
        age_bins = torch.arange(0, age_logits.size(1), device=DEVICE)
        predicted_age = (age_probs * age_bins).sum(dim=1).item()

        # Verification Embedding
        embedding = verif_model(verif_input)

    return {
        "box": box.astype(int).tolist(),  # [x1, y1, x2, y2]
        "age": predicted_age,
        "embedding": embedding,
        "original_pil": img_pil,
    }


def create_visualization(
    res1: dict, res2: dict, similarity: float, is_match: bool, save_path: Path
):
    """
    Creates a side-by-side composite image with annotations using OpenCV.
    """

    def prepare_img(res):
        # Convert PIL to CV2 (BGR)
        img = np.array(res["original_pil"])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw Box
        x1, y1, x2, y2 = res["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw Age Background and Text
        label = f"Age: {res['age']:.1f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img, (x1, y1 - 35), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(
            img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
        )

        return img

    img1 = prepare_img(res1)
    img2 = prepare_img(res2)

    # Resize images to match height for clean concatenation
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2, 600)  # Minimum 600px height

    def resize_h(img, target_h):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h))

    img1 = resize_h(img1, target_h)
    img2 = resize_h(img2, target_h)

    # Concatenate horizontally
    combined = np.hstack([img1, img2])

    # Add Top Border for Result Title
    border_h = 80
    combined = cv2.copyMakeBorder(
        combined, border_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # Add Title Text
    status = "SAME PERSON" if is_match else "DIFFERENT PEOPLE"
    color = (0, 200, 0) if is_match else (0, 0, 255)  # Green or Red
    text = f"{status} (Score: {similarity:.2f})"

    # Center text
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    cx = combined.shape[1] // 2 - tw // 2
    cv2.putText(combined, text, (cx, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Save
    cv2.imwrite(str(save_path), combined)
    print(f"Saved visualization to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Age Estimation & Face Verification")
    parser.add_argument("img1", type=str, help="Path to first image")
    parser.add_argument("img2", type=str, help="Path to second image")
    parser.add_argument("output_folder", type=str, help="Folder to save results")
    parser.add_argument("--json", action="store_true", help="Save detailed JSON output")
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to Age Model checkpoint"
    )

    args = parser.parse_args()

    # 1. Resolve Checkpoint
    ckpt_path = args.ckpt
    if ckpt_path is None:
        if BEST_MODEL_PATH and Path(BEST_MODEL_PATH).exists():
            ckpt_path = BEST_MODEL_PATH
        else:
            print(
                "Error: No checkpoint provided and config.BEST_MODEL_PATH is invalid."
            )
            return

    # 2. Setup Output
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate a filename base
    name1 = Path(args.img1).stem
    name2 = Path(args.img2).stem
    base_name = f"{name1}_vs_{name2}"

    # 3. Load Models
    mtcnn, age_model, verif_model = load_models(ckpt_path)
    age_transforms = get_age_transforms()

    # 4. Process Images
    print("Processing images...")
    res1 = process_single_image(
        args.img1, mtcnn, age_model, verif_model, age_transforms
    )
    res2 = process_single_image(
        args.img2, mtcnn, age_model, verif_model, age_transforms
    )

    if res1 is None or res2 is None:
        print("Aborting: Could not detect faces in one or both images.")
        return

    # 5. Verify (Cosine Similarity)
    emb1 = res1["embedding"]
    emb2 = res2["embedding"]

    similarity = nn.functional.cosine_similarity(emb1, emb2).item()
    is_match = similarity > VERIFICATION_THRESHOLD

    print("-" * 30)
    print(f"Img 1 Age: {res1['age']:.1f}")
    print(f"Img 2 Age: {res2['age']:.1f}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Verdict: {'MATCH' if is_match else 'NO MATCH'}")
    print("-" * 30)

    # 6. Save Visualization
    vis_path = out_dir / f"{base_name}_result.jpg"
    create_visualization(res1, res2, similarity, is_match, vis_path)

    # 7. Save JSON (Optional)
    if args.json:
        json_path = out_dir / f"{base_name}_data.json"
        data = {
            "image_1": str(args.img1),
            "image_2": str(args.img2),
            "face_1": {"age_prediction": round(res1["age"], 2), "bbox": res1["box"]},
            "face_2": {"age_prediction": round(res2["age"], 2), "bbox": res2["box"]},
            "verification": {
                "is_match": bool(is_match),
                "similarity_score": round(similarity, 4),
                "threshold_used": VERIFICATION_THRESHOLD,
            },
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved JSON data to: {json_path}")


if __name__ == "__main__":
    main()
