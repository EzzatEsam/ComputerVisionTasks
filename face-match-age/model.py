import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import lightning as pl


class AgeEstimationModel(pl.LightningModule):
    def __init__(self, num_classes=101, pretrained="vggface2", freeze_backbone=False):
        """
        Args:
            num_classes: Number of age bins (e.g., 101 for ages 0-100)
            pretrained: 'vggface2', 'casia-webface', or None
            freeze_backbone: If True, only train the final layers
        """
        super(AgeEstimationModel, self).__init__()

        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        input_dim = 512

        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),  # Final output: 101 age probabilities
        )

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mae_loss = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x

    @staticmethod
    def _transform_ages_to_distribution(
        ages: torch.Tensor, num_classes=101, sigma=2
    ) -> torch.Tensor:
        """
        Convert ages to a Gaussian distribution over age bins.

        Args:
            ages: Tensor of shape (batch_size,) with integer ages
            num_classes: Number of age bins
            sigma: Standard deviation for Gaussian

        Returns:
            Tensor of shape (batch_size, num_classes) with age distributions
        """
        batch_size = ages.size(0)
        age_bins = (
            torch.arange(0, num_classes)
            .reshape(1, -1)
            .repeat(batch_size, 1)
            .to(ages.device)
        )  # (batch_size, num_classes)
        ages = ages.reshape(-1, 1).repeat(1, num_classes)  # (batch_size, num_classes)

        distributions = torch.exp(
            -0.5 * ((age_bins - ages) / sigma) ** 2
        )  # Gaussian formula
        distributions = distributions / distributions.sum(dim=1, keepdim=True)

        return distributions

    def training_step(self, batch, batch_idx):
        (images, ages) = batch
        logits = self.forward(images)

        age_distributions = self._transform_ages_to_distribution(ages)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        loss = self.kl_loss(log_probs, age_distributions)
        pred_ages = (
            logits.softmax(dim=1) * torch.arange(0, logits.size(1)).to(logits.device)
        ).sum(dim=1)
        mae = self.mae_loss(pred_ages, ages)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (images, ages) = batch
        logits = self.forward(images)

        age_distributions = self._transform_ages_to_distribution(ages)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        loss = self.kl_loss(log_probs, age_distributions)
        pred_ages = (
            logits.softmax(dim=1) * torch.arange(0, logits.size(1)).to(logits.device)
        ).sum(dim=1)
        mae = self.mae_loss(pred_ages, ages)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
