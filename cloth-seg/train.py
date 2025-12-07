import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from model import ClothSegmentationModel
from loader import train_loader, val_loader
from callbacks import ImageLogger
import argparse


def train(last_ckpt=None):
    # Initialize model
    model = ClothSegmentationModel()

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="cloth-segmentation-{epoch:02d}-{val_loss:.4f}-{val_iou:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_iou",
        patience=6,
        mode="max",
        verbose=True,
    )

    image_logger_callback = ImageLogger(num_samples=3)

    # Setup logger
    logger = TensorBoardLogger("logs", name="cloth_segmentation")

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=15,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, image_logger_callback],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt)

    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Path to checkpoint to resume"
    )
    args = parser.parse_args()

    train(last_ckpt=args.ckpt)
