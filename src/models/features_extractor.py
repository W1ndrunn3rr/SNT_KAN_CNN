import lightning as L
import torch
import torch.nn as nn


class FeaturesExtractor(L.LightningModule):
    def __init__(
        self,
        channels=[64, 128, 256, 512, 512],
        output_dim=256,
        dropout=0.5,
        classification: int = False,
        num_classes: int = 6,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_ch = 3
        for out_ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    (
                        nn.MaxPool2d(2, 2)
                        if out_ch != channels[-1]
                        else nn.AdaptiveAvgPool2d((1, 1))
                    ),
                ]
            )
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(channels[-1], output_dim)
        )

        self.classification_layer = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        if self.hparams.classification:
            x = self.classification_layer(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    @classmethod
    def load_from_universal_cnn_checkpoint(cls, checkpoint_path: str, **kwargs):

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model.") and not key.startswith("model.model"):
                new_key = key[6:]
                new_state_dict[new_key] = value

        instance = cls(**kwargs)

        instance.load_state_dict(new_state_dict, strict=False)

        return instance
