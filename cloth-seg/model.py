import segmentation_models_pytorch as smp
import lightning as L
import torch
import torchmetrics
from config import ENCODER_NAME, ENCODER_WEIGHTS, NUM_CLASSES
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassF1Score,
    MulticlassAccuracy,
)


class ClothSegmentationModel(L.LightningModule):
    def __init__(self, freeze_encoder: bool = False):
        """
        Initialize the ClothSegmentationModel.

        Args:
            freeze_encoder (bool, optional): Freeze the weights of encoder. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=NUM_CLASSES,
        )

        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True, smooth=1
        )
        self.pixel_loss = smp.losses.FocalLoss(mode="multiclass")


        metrics = torchmetrics.MetricCollection(
            {
                "val_iou": MulticlassJaccardIndex(
                    num_classes=NUM_CLASSES,
                    average="macro",  # Computes metric for each class and averages them equally
                ),
                "val_f1": MulticlassF1Score(num_classes=NUM_CLASSES, average="macro"),
                "val_acc": MulticlassAccuracy(
                    num_classes=NUM_CLASSES,
                    average="micro",  # Overall pixel accuracy
                ),
            }
        )
        self.val_metrics = metrics.clone()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str):
        images, masks = batch
        outputs = self(images)

        dice_loss = self.dice_loss(outputs, masks)
        pixel_loss = self.pixel_loss(outputs, masks)
        loss = 0.5 * dice_loss + 0.5 * pixel_loss

        self.log(f"{stage}_dice_loss", dice_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_pixel_loss", pixel_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss, outputs, masks

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, masks = self._shared_step(batch, stage="val")

        preds = torch.argmax(outputs, dim=1)

        self.val_metrics.update(preds, masks)

        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
