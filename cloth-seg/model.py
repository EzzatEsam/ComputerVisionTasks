import segmentation_models_pytorch as smp
import lightning as L
import torch
from config import ENCODER_NAME, ENCODER_WEIGHTS, NUM_CLASSES
import matplotlib.pyplot as plt

class ClothSegmentationModel(L.LightningModule):
    def __init__(self, freeze_encoder: bool = False):
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
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.pixel_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str):
        images, masks = batch
        outputs = self(images)
        dice_loss = self.dice_loss(outputs, masks)
        pixel_loss = self.pixel_loss(outputs, masks)
        loss = 0.5 * dice_loss + 0.5 * pixel_loss
        self.log(f"{stage}_dice_loss", dice_loss ,on_step=True , on_epoch=True)
        self.log(f"{stage}_pixel_loss", pixel_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True , prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="train")
        if batch_idx % 100 == 0:
            self.visualize_predictions(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    
    
    def visualize_predictions(self, batch):
        images, masks = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        
        # Take first image from batch for visualization
        img = images[0].cpu().detach()
        mask_true = masks[0].cpu().detach().numpy()
        mask_pred = preds[0].cpu().detach().numpy()
        
        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")
        
        # Ground truth mask
        axes[1].imshow(mask_true, cmap="tab20", vmin=0, vmax=NUM_CLASSES-1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        # Predicted mask
        axes[2].imshow(mask_pred, cmap="tab20", vmin=0, vmax=NUM_CLASSES-1)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # Log to TensorBoard
        if self.logger is not None:
            tb = self.logger.experiment
            tb.add_figure(
                "predictions",
                fig,
                global_step=self.global_step
            )
        
        plt.close(fig)