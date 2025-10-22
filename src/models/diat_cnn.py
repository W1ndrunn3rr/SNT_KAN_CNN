import torch
import torch.nn as nn


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.branch_a = ConvRelu(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0
        )

        self.reduce = ConvRelu(
            in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0
        )

        self.branch_b = nn.Sequential(
            ConvRelu(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.main = nn.Sequential(
            ConvRelu(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_a = self.branch_a(x)
        main_branch = self.reduce(x)
        branch_b = self.branch_b(main_branch)
        main_branch = self.main(main_branch)

        return main_branch + branch_a + branch_b


class DiatNet(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes

        self.feature_extractor = nn.Sequential(
            ConvRelu(in_channels=3, out_channels=96, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block1 = ResidualBlock(in_channels=96, mid_channels=32, out_channels=64)

        self.block2 = ResidualBlock(in_channels=64, mid_channels=64, out_channels=128)

        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.block3 = ResidualBlock(in_channels=128, mid_channels=96, out_channels=256)

        self.block4 = ResidualBlock(in_channels=256, mid_channels=128, out_channels=512)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x
