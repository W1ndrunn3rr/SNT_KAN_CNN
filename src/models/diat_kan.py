from .features_extractor import FeaturesExtractor
import lightning as L
import torch
from kan import KAN
from typing import Literal
from fastkan import FastKAN
import torch.nn.functional as F
from torchmetrics import Accuracy


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
    ):
        super(DIATKAN, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.flatten_size = flatten_size

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

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
        self.val_acc(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)
        self.test_conf_matrix.update(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
