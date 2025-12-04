import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from model import ClothSegmentationModel
from loader import train_loader, val_loader


def train():
    # Initialize model
    model = ClothSegmentationModel()

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="cloth-segmentation-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    # Setup logger
    logger = TensorBoardLogger("logs", name="cloth_segmentation")

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        # precision="16-mixed",  # Use mixed precision for faster training
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    print("Training completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
