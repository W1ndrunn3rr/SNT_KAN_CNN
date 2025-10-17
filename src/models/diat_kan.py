from .features_extractor import FeaturesExtractor
import lightning as L
import torch
from kan import KAN
from typing import Literal
from fastkan import FastKAN
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from PIL import Image


class DIATKAN(L.LightningModule):
    def __init__(
        self,
        model_type: Literal["NORMAL", "FAST"],
        num_classes: int = 6,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        kan_width: list[int] = [128, 64, 32],
        grid_size: int = 5,
        k: int = 3,
        flatten_size: int = 256,
        class_names: list[str] = None,
    ):
        super(DIATKAN, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.flatten_size = flatten_size
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Confusion Matrix
        self.val_conf_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )
        self.test_conf_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        # Store predictions for plotting
        self.validation_step_outputs = []

        self.feature_extractor = FeaturesExtractor()

        self.kan_layers = [self.flatten_size] + kan_width + [num_classes]

        self.kan = (
            KAN(width=self.kan_layers, grid=grid_size, k=k, seed=42)
            if model_type == "NORMAL"
            else FastKAN(layers_hidden=self.kan_layers, num_grids=grid_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.kan(x)
        return x

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_accuracy", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        # Update metrics
        self.val_acc(y_hat, y)
        self.val_conf_matrix.update(preds, y)

        # Store outputs for epoch end
        self.validation_step_outputs.append(
            {"loss": loss.detach(), "preds": preds.detach(), "targets": y.detach()}
        )

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """Log confusion matrix at the end of validation epoch"""
        if len(self.validation_step_outputs) == 0:
            return

        # Compute confusion matrix
        conf_matrix = self.val_conf_matrix.compute()

        # Calculate per-class accuracy
        per_class_acc = conf_matrix.diag() / conf_matrix.sum(dim=1)

        # Log per-class accuracy
        for idx, class_name in enumerate(self.class_names):
            self.log(
                f"Validation/Accuracy_{class_name}", per_class_acc[idx], on_epoch=True
            )

        # Create figure
        fig = self._plot_confusion_matrix(conf_matrix, "Validation Confusion Matrix")

        # Log to TensorBoard
        self.logger.experiment.add_figure(
            "Validation/Confusion Matrix", fig, global_step=self.current_epoch
        )

        plt.close(fig)

        # Reset for next epoch
        self.val_conf_matrix.reset()
        self.validation_step_outputs.clear()

    def _plot_confusion_matrix(self, conf_matrix, title):
        """Create a confusion matrix plot"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Convert to numpy and normalize
        cm = conf_matrix.cpu().numpy()
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={"label": "Percentage"},
        )

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)

        # Add counts as text
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                count = cm[i, j]
                if count > 0:
                    ax.text(
                        j + 0.5,
                        i + 0.7,
                        f"({int(count)})",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="gray",
                    )

        plt.tight_layout()
        return fig

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        self.test_acc(y_hat, y)
        self.test_conf_matrix.update(preds, y)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_acc, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        """Log confusion matrix at the end of test epoch"""
        conf_matrix = self.test_conf_matrix.compute()

        fig = self._plot_confusion_matrix(conf_matrix, "Test Confusion Matrix")

        self.logger.experiment.add_figure("Test/Confusion Matrix", fig, global_step=0)

        plt.close(fig)
        self.test_conf_matrix.reset()

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
