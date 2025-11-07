from .features_extractor import FeaturesExtractor
import lightning as L
import torch
import torchvision
from fastkan import FastKAN
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os


class UniversalCNN(L.LightningModule):
    def __init__(
        self,
        model_type: str = "KAN_FAST",
        num_classes: int = 6,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        kan_width: list[int] = [256, 128, 64, 32, 16],
        grid_size: int = 8,
        k: int = 3,
        flatten_size: int = 3 * 224 * 224,
        class_names: list[str] = None,
        scheduler_config: dict | None = None,
        feature_extractor_path="models/kan_fast_feature_extractor.pth",
    ):
        super(UniversalCNN, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.flatten_size = flatten_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        if scheduler_config is None:
            scheduler_config = {}
        elif hasattr(scheduler_config, "items"):
            scheduler_config = dict(scheduler_config)
        self.scheduler_config = scheduler_config

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_conf_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )
        self.test_conf_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

        self.validation_step_outputs = []

        if model_type == "just_kan":
            self.kan_layers = [self.flatten_size] + kan_width + [num_classes]
            self.model = torch.nn.Sequential(
                torch.nn.Flatten(),
                FastKAN(layers_hidden=self.kan_layers, num_grids=grid_size),
            )

        elif model_type in ["KAN_FAST", "KAN", "FAST"]:
            pretrained_path = (
                feature_extractor_path if model_type == "KAN_FAST" else None
            )

            if pretrained_path and os.path.exists(pretrained_path):
                try:
                    if pretrained_path.endswith(".ckpt"):
                        self.feature_extractor = (
                            FeaturesExtractor.load_from_universal_cnn_checkpoint(
                                pretrained_path,
                                num_classes=self.num_classes,
                                classification=False,
                            )
                        )
                        print(f"Loaded pretrained weights from {pretrained_path}")
                        self.feature_extractor.freeze()
                    else:
                        self.feature_extractor = FeaturesExtractor(
                            num_classes=self.num_classes, classification=False
                        )
                        self.feature_extractor.load_state_dict(
                            torch.load(
                                pretrained_path, map_location="cpu", weights_only=False
                            )
                        )
                        print(f"Loaded pretrained weights from {pretrained_path}")
                        self.feature_extractor.freeze()
                except Exception as e:
                    print(f"Warning: Could not load pretrained weights: {e}")
                    print("Initializing feature extractor with random weights")
                    self.feature_extractor = FeaturesExtractor(
                        num_classes=self.num_classes, classification=False
                    )
            else:
                self.feature_extractor = FeaturesExtractor(
                    num_classes=self.num_classes, classification=False
                )

            self.kan_layers = [self.flatten_size] + kan_width + [num_classes]

            self.model = torch.nn.Sequential(
                self.feature_extractor,
                torch.nn.BatchNorm1d(self.flatten_size),
                torch.nn.Dropout(0.2),
                FastKAN(layers_hidden=self.kan_layers, num_grids=grid_size),
            )

        elif model_type == "resnet50":
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.model = torchvision.models.resnet50(weights=weights)
            num_ftrs = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

        elif model_type == "vgg16":
            weights = torchvision.models.VGG16_Weights.DEFAULT
            self.model = torchvision.models.vgg16(weights=weights)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, num_classes)

        elif model_type == "densenet121":
            weights = torchvision.models.DenseNet121_Weights.DEFAULT
            self.model = torchvision.models.densenet121(weights=weights)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(num_ftrs, num_classes)

        elif model_type == "mobilenet_v2":
            weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
            self.model = torchvision.models.mobilenet_v2(weights=weights)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

        elif model_type == "efficientnet_b0":
            weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            self.model = torchvision.models.efficientnet_b0(weights=weights)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

        elif model_type == "vit_b_16":
            weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            self.model = torchvision.models.vit_b_16(weights=weights)
            num_ftrs = self.model.heads.head.in_features
            self.model.heads.head = torch.nn.Linear(num_ftrs, num_classes)

        elif model_type == "features_extractor":
            self.model = FeaturesExtractor(classification=True, num_classes=num_classes)

        elif model_type == "diatnet":
            from .diat_cnn import DiatNet

            self.model = DiatNet(num_classes=num_classes)

        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Supported: KAN_FAST, resnet18, resnet50, vgg16, densenet121, mobilenet_v2, efficientnet_b0, vit_b_16, features_extractor, diatnet"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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

        self.val_acc(y_hat, y)
        self.val_conf_matrix.update(preds, y)

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

        conf_matrix = self.val_conf_matrix.compute()

        per_class_acc = conf_matrix.diag() / conf_matrix.sum(dim=1)

        for idx, class_name in enumerate(self.class_names):
            self.log(
                f"Validation/Accuracy_{class_name}", per_class_acc[idx], on_epoch=True
            )

        fig = self._plot_confusion_matrix(conf_matrix, "Validation Confusion Matrix")

        self.logger.experiment.add_figure(
            "Validation/Confusion Matrix", fig, global_step=self.current_epoch
        )

        plt.close(fig)

        self.val_conf_matrix.reset()
        self.validation_step_outputs.clear()

    def _plot_confusion_matrix(self, conf_matrix, title):
        """Create a confusion matrix plot"""
        fig, ax = plt.subplots(figsize=(10, 8))

        cm = conf_matrix.cpu().numpy()

        row_sums = cm.sum(axis=1)[:, np.newaxis]
        row_sums[row_sums == 0] = 1
        cm_normalized = cm.astype("float") / row_sums

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
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        factor = self.scheduler_config.get("factor", 0.5)
        patience = self.scheduler_config.get("patience", 3)
        min_lr = self.scheduler_config.get("min_lr", 1e-7)
        monitor = self.scheduler_config.get("monitor", "val_loss")
        mode = self.scheduler_config.get("mode", "min")
        threshold = self.scheduler_config.get("threshold", 1e-4)
        cooldown = self.scheduler_config.get("cooldown", 0)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            threshold=threshold,
            cooldown=cooldown,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor,
                "interval": "epoch",
                "frequency": 1,
            },
        }
