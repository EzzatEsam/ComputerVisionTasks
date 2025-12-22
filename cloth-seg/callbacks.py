import lightning as L
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import CATEGORIES  # Import your categories


class ImageLogger(L.Callback):
    def __init__(self, num_samples=3, train_plot_every_n_batches=200):
        super().__init__()
        self.num_samples = num_samples
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.train_plot_every_n_batches = train_plot_every_n_batches

        # Create a reverse mapping: ID -> Name (e.g., 2 -> "upperbody")
        self.id_to_name = {v: k for k, v in CATEGORIES.items()}
        
        # Get the colormap for consistent colors
        self.cmap = plt.cm.get_cmap('tab20', len(self.id_to_name) + 1)

    def _visualize_batch(self, trainer, pl_module, batch, stage: str):
        images, masks = batch

        images = images[: self.num_samples].cpu()
        masks = masks[: self.num_samples].cpu()

        # 1. Capture state BEFORE changing it
        was_training = pl_module.training
        pl_module.eval()

        with torch.no_grad():
            logits = pl_module(images.to(pl_module.device))
            preds = torch.argmax(logits, dim=1).cpu()

        # 2. Restore state
        pl_module.train(was_training)

        # Create plot with extra space for legend
        fig, axes = plt.subplots(len(images), 3, figsize=(25, 8 * len(images)))

        for i in range(len(images)):
            # Denormalize
            img = images[i] * self.std + self.mean
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()

            # Handle axes (if batch=1, axes is 1D array)
            ax = axes[i] if len(images) > 1 else axes

            # --- Input Image ---
            ax[0].imshow(img)
            ax[0].set_title("Input")
            ax[0].axis("off")

            # --- Ground Truth ---
            ax[1].imshow(
                masks[i],
                cmap='tab20',
                interpolation="nearest",
                vmin=0,
                vmax=len(self.id_to_name),
            )
            ax[1].set_title("Ground Truth")
            ax[1].axis("off")

            # Extract unique classes present in GT and create legend
            gt_ids = torch.unique(masks[i]).numpy()
            gt_patches = []
            for idx in gt_ids:
                color = self.cmap(idx)
                label = self.id_to_name.get(idx, f"Unk:{idx}")
                gt_patches.append(mpatches.Patch(color=color, label=label))
            
            ax[1].legend(
                handles=gt_patches,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=min(3, len(gt_patches)),
                fontsize=9,
                frameon=True
            )

            # --- Prediction ---
            ax[2].imshow(
                preds[i],
                cmap='tab20',
                interpolation="nearest",
                vmin=0,
                vmax=len(self.id_to_name),
            )
            ax[2].set_title("Prediction")
            ax[2].axis("off")

            # Extract unique classes present in Prediction and create legend
            pred_ids = torch.unique(preds[i]).numpy()
            pred_patches = []
            for idx in pred_ids:
                color = self.cmap(idx)
                label = self.id_to_name.get(idx, f"Unk:{idx}")
                pred_patches.append(mpatches.Patch(color=color, label=label))
            
            ax[2].legend(
                handles=pred_patches,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=min(3, len(pred_patches)),
                fontsize=9,
                frameon=True
            )

        plt.tight_layout()

        # Send to TensorBoard
        if trainer.logger:
            trainer.logger.experiment.add_figure(
                f"{stage}/Predictions", fig, global_step=trainer.global_step
            )

        plt.close(fig)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.train_plot_every_n_batches != 0:
            return
        self._visualize_batch(trainer, pl_module, batch, stage="Train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx != 0:
            return
        self._visualize_batch(trainer, pl_module, batch, stage="Validation")