import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from model import AgeEstimationModel
from loader import train_loader, val_loader
from callbacks import ImageLogger
import argparse


def train(last_ckpt=None):
    # Initialize model
    model = AgeEstimationModel()

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="face-age-det{epoch:02d}-{val_loss:.4f}-{val_mae:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=6,
        mode="min",
        verbose=True,
    )

    image_logger_callback = ImageLogger(
        train_log_every_n_steps=50, val_log_every_n_steps=50, max_images=4
    )

    # Setup logger
    logger = TensorBoardLogger("logs", name="face_age_det")

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=25,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop_callback, image_logger_callback],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.25,
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
