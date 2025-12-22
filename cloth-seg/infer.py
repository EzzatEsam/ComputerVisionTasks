import argparse
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms

from model import ClothSegmentationModel
from config import IMG_SIZE, CATEGORIES, BEST_MODEL_PATH, MEAN, STD


ID2LABEL = {v: k for k, v in CATEGORIES.items()}

np.random.seed(42)
COLORS = np.array(
    [
        [0, 0, 0],  # 0: background (Black)
        [255, 0, 0],  # 1: closures (Blue)
        [0, 255, 0],  # 2: upperbody (Green)
        [0, 0, 255],  # 3: head (Red)
        [255, 255, 0],  # 4: legs and feet (Cyan)
        [255, 0, 255],  # 5: garment parts (Magenta)
        [0, 255, 255],  # 6: arms and hands (Yellow)
        [128, 0, 0],  # 7: decorations
        [0, 128, 0],  # 8: neck
        [0, 0, 128],  # 9: others
        [128, 128, 0],  # 10: lowerbody
        [128, 0, 128],  # 11: waist
        [0, 128, 128],  # 12: wholebody
    ]
)


def get_transform() -> transforms.Compose:
    """
    Constructs the torchvision preprocessing transform.
    Matches ImageNet statistics expected by the encoder.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def mask_to_rle(binary_mask: np.ndarray) -> list[int]:
    """
    Convert a binary mask to RLE (Run-Length Encoding).
    """
    pixels = binary_mask.flatten()
    # We add 0 at start and end to ensure runs logic handles edges correctly
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()


def load_model(ckpt_path: str) -> tuple[ClothSegmentationModel, torch.device]:
    """Loads the Lightning model from checkpoint."""
    print(f"Loading model from: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = ClothSegmentationModel.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    return model, device


def predict(
    image_path: Path,
    model: ClothSegmentationModel,
    device: torch.device,
    transform: transforms.Compose,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads image, preprocesses, runs inference, and resizes mask to original size.
    """
    # 1. Load Image
    original_image_bgr = cv2.imread(str(image_path))
    if original_image_bgr is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    # Convert BGR to RGB for the model
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = original_image_rgb.shape

    # 2. Preprocess
    input_tensor = transform(original_image_rgb)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dim -> [1, C, H, W]

    # 3. Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_mask_tensor = torch.argmax(probs, dim=1).squeeze()  # [H, W]

    # 4. Resize mask back to original image size
    pred_mask_small = pred_mask_tensor.cpu().numpy().astype(np.uint8)
    pred_mask_original = cv2.resize(
        pred_mask_small, (w, h), interpolation=cv2.INTER_NEAREST
    )

    return original_image_bgr, pred_mask_original

def draw_legend_sidebar(
    image: np.ndarray, 
    unique_classes: np.ndarray, 
) -> np.ndarray:
    """
    Appends a white sidebar to the right of the image with a legend.
    """
    h, w, _ = image.shape
    
    # sidebar width: 250px or 25% of width, whichever is larger, to ensure text fits
    sidebar_w = max(250, int(w * 0.25))
    
    # Create a white canvas for the sidebar
    sidebar = np.ones((h, sidebar_w, 3), dtype=np.uint8) * 255
    
    # Layout settings
    start_y = 30
    gap = 40
    box_size = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0, 0, 0) # Black text
    unique_classes = unique_classes[unique_classes != 0]  # Exclude background if needed
    for i, class_id in enumerate(unique_classes):

        # Get color (ensure it matches the format used in the mask)
        # Assuming COLORS are RGB, we convert to BGR for OpenCV
        color_rgb = COLORS[class_id]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        
        # Get Name
        label = ID2LABEL.get(class_id, f"Class {class_id}")
        
        # Draw the color box
        y_pos = start_y + (i * gap)
        p1 = (15, y_pos)
        p2 = (15 + box_size, y_pos + box_size)
        cv2.rectangle(sidebar, p1, p2, color_bgr, -1)
        
        # Draw the text
        text_pos = (15 + box_size + 10, y_pos + box_size - 5)
        cv2.putText(sidebar, label, text_pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Combine original image and sidebar horizontally
    combined = np.hstack([image, sidebar])
    return combined

def save_visualizations(
    image_bgr: np.ndarray, 
    mask: np.ndarray, 
    output_dir: Path, 
    file_stem: str,
) -> tuple[Path, Path]:
    """
    Saves the raw mask (color coded) and the overlay image with a legend.
    """
    # 1. Create Colored Mask
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    unique_classes = np.unique(mask)
    for class_id in unique_classes:
        if class_id == 0:
            continue  # Skip background
        color_mask[mask == class_id] = COLORS[class_id]

    # 2. Create Overlay (50% opacity)
    # Note: image_bgr is BGR, color_mask is RGB (from our random gen)
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.6, color_mask_bgr, 0.4, 0)

    # --- NEW STEP: Add Legend to Overlay ---
    overlay_with_legend = draw_legend_sidebar(
        overlay, unique_classes
    )

    # Save Colored Mask (usually we keep this raw without legend for analysis)
    mask_path = output_dir / f"{file_stem}_mask.png"
    cv2.imwrite(str(mask_path), color_mask_bgr)

    # Save Overlay (with legend for visualization)
    overlay_path = output_dir / f"{file_stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay_with_legend)

    return mask_path, overlay_path


def generate_json_output(mask: np.ndarray, output_dir: Path, file_stem: str) -> Path:
    """
    Generates JSON with RLE masks for detected objects.
    """
    found_classes = np.unique(mask)
    segments: list[dict] = []

    for class_id in found_classes:
        if class_id == 0:
            continue

        # Create binary mask for this specific class
        class_binary_mask = (mask == class_id).astype(np.uint8)

        segment_info = {
            "class_id": int(class_id),
            "label": ID2LABEL.get(int(class_id), "unknown"),
            "rle": mask_to_rle(class_binary_mask),
        }
        segments.append(segment_info)

    payload = {"file": file_stem, "segments": segments}

    json_path = output_dir / f"{file_stem}_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f)

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloth Segmentation Inference CLI")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--output", type=str, default="output", help="Directory to save results"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(BEST_MODEL_PATH),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--json", action="store_true", help="Generate JSON with RLE masks"
    )

    args = parser.parse_args()

    # Setup paths
    img_path = Path(args.image)
    output_dir = Path(args.output)

    if not img_path.exists():
        print(f"Error: Input image not found at {img_path}")
        return
    if output_dir.exists() and not output_dir.is_dir():
        print(f"Error: Output path {output_dir} exists and is not a directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify Checkpoint Exists
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Please train the model first or provide valid path via --ckpt")
        return

    # Load Model & Transform
    try:
        model, device = load_model(str(ckpt_path))
        transform = get_transform()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Processing: {img_path.name}")

    try:
        # Run Inference
        original_img, mask = predict(img_path, model, device, transform)

        # Save Visuals
        m_path, o_path = save_visualizations(
            original_img, mask, output_dir, img_path.stem
        )
        print(f"Saved Mask: {m_path}")
        print(f"Saved Overlay: {o_path}")

        # Save JSON (Optional)
        if args.json:
            j_path = generate_json_output(mask, output_dir, img_path.stem)
            print(f"Saved JSON: {j_path}")

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
