import lightning as L
import torch
import matplotlib.pyplot as plt

from config import TRANSFORM_MEAN, TRANSFORM_STD


class ImageLogger(L.Callback):
    def __init__(
        self, train_log_every_n_steps=50, val_log_every_n_steps=50, max_images=4
    ):
        super().__init__()
        self.train_log_every_n_steps = train_log_every_n_steps
        self.val_log_every_n_steps = val_log_every_n_steps
        self.max_images = max_images

    def _visualize_batch(self, trainer, pl_module, batch, stage: str):
        imgs, ages = batch
        imgs = imgs[: self.max_images]
        ages = ages[: self.max_images]

        # Get predictions
        was_training = pl_module.training
        pl_module.eval()

        with torch.no_grad():
            logits = pl_module(imgs.to(pl_module.device))
            props = torch.softmax(logits, dim=1)
            all_ages = torch.arange(0, logits.size(1)).to(logits.device)
            preds = (props * all_ages).sum(dim=1)

        # 2. Restore state
        pl_module.train(was_training)

        # 3. Create figure
        fig, axes = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4))
        if len(imgs) == 1:
            axes = [axes]
        for img, age, pred, ax in zip(imgs, ages, preds, axes):
            img = img.cpu()
            img = img * torch.tensor(TRANSFORM_STD).view(3, 1, 1) + torch.tensor(
                TRANSFORM_MEAN
            ).view(3, 1, 1)

            # Display predicted and true ages
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
            ax.set_title(
                f"True: {age.item():.2f}\nPred: {pred.item():.2f}",
                color="blue",
                fontsize=14,
            )
            ax.axis("off")

        if trainer.logger:
            trainer.logger.experiment.add_figure(
                f"{stage}_images", fig, global_step=trainer.global_step
            )

        plt.close(fig)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx % self.train_log_every_n_steps) == 0:
            self._visualize_batch(trainer, pl_module, batch, stage="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (batch_idx % self.val_log_every_n_steps) == 0:
            self._visualize_batch(trainer, pl_module, batch, stage="val")
