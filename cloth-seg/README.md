# Cloth Segmentation with PyTorch Lightning & SMP

This project implements a semantic segmentation pipeline for clothing analysis using the **Fashionpedia** dataset. It utilizes `segmentation-models-pytorch` (U-Net architecture with a MIT-B1 encoder) and `Lightning` for training orchestration.

## Prerequisites

This project is managed using **uv**. Ensure you have it installed:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

1. **Initialize the environment and sync dependencies:**

   Run:
   ```bash
   uv sync
   ```

## Usage Guide

## Configuration (Optional)

You can customize the training and model behavior by modifying `config.py`. The key parameters are:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `BATCH_SIZE` | `32` | Reduce this to `16` or `8` if you run into CUDA Out-of-Memory errors. |
| `IMG_SIZE` | `512` | The input resolution for the model. Higher values (e.g., `1024`) capture more detail but require more VRAM and training time. |
| `ENCODER_NAME` | `"mit_b1"` | The backbone architecture. You can switch this to other encoders supported by SMP (e.g., `"resnet34"`, `"efficientnet-b0"`, `"mit_b2"`). |
| `ENCODER_WEIGHTS` | `"imagenet"` | Pre-training weights. Use `None` to train from scratch. |
| `DATA_PATH` | `"data"` | The root directory where datasets are downloaded and extracted. |


### 1. Data Preparation
Before training, you must download the dataset and convert the COCO annotations into segmentation masks.

Run the data preparation script:
```bash
uv run data_prep.py
```
*   **What this does:** Downloads ~2GB of data from S3, extracts it to `data/fashionist`, and processes vector annotations into PNG masks.

### 2. Visualization (Optional)
To verify that data is loading correctly and masks align with images:

```bash
uv run loader.py
```
*   This will save visualization samples to the `visualized_samples/` directory.

### 3. Training
Start the training process. The model uses a U-Net with a pre-trained `mit_b1` encoder.

```bash
uv run train.py
```

*   **Checkpoints:** Saved automatically to `checkpoints/` based on validation loss.
*   **Logging:** Metrics and prediction images are logged to `logs/`.
*   **Resume Training:** To resume from a specific checkpoint:
    ```bash
    uv run train.py --ckpt checkpoints/last.ckpt
    ```

### 4. Monitor Training
You can view loss curves, metrics (IoU, F1), and validation prediction samples using TensorBoard:

```bash
uv run tensorboard --logdir logs
```

### 5. Inference
To run the model on a new image and generate predictions:

```bash
uv run infer.py path/to/your/image.jpg
```

**Arguments:**
*   `--ckpt`: Path to model checkpoint (defaults to path in `config.py`).
*   `--output`: Directory to save results (default: `output/`).
*   `--json`: If set, generates a JSON file containing RLE (Run-Length Encoding) for detected segments.

**Example:**
```bash
uv run infer.py test_image.jpg --output results --json
```

### Results
Results can be found in dedicated report file:
[REPORT.md](REPORT.md)

## Project Structure

```text
.
├── callbacks.py      # Custom Lightning callback for logging image predictions
├── config.py         # Hyperparameters, file paths, and label definitions
├── data_prep.py      # Script to download data and generate masks
├── infer.py          # CLI tool for running inference on new images
├── loader.py         # Dataset class (Torchvision) and DataLoader setup
├── model.py          # U-Net Model definition (LightningModule)
├── train.py          # Main training script
└── README.md         # This file
```

## Classes & Labels
The model segments the following 13 categories:
0. Background
1. Closures
2. Upperbody
3. Head
4. Legs and Feet
5. Garment Parts
6. Arms and Hands
7. Decorations
8. Neck
9. Others
10. Lowerbody
11. Waist
12. Wholebody