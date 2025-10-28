import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


def _compute_mean_std(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> tuple[list[float], list[float]]:
    dataset = ImageFolder(
        root=data_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    channel_sum = torch.zeros(3, dtype=torch.double)
    channel_sq_sum = torch.zeros(3, dtype=torch.double)
    total_pixels = 0

    for images, _ in loader:
        images = images.to(dtype=torch.double)
        batch_size_curr, channels, height, width = images.shape
        pixels = batch_size_curr * height * width
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images**2).sum(dim=(0, 2, 3))
        total_pixels += pixels

    if total_pixels == 0:
        raise ValueError(f"Dataset at {data_dir} is empty. Cannot compute mean/std.")

    mean = channel_sum / total_pixels
    var = (channel_sq_sum / total_pixels) - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    return mean.float().tolist(), std.float().tolist()


class DataProcessor(nn.Module):
    def __init__(
        self,
        data_dir: str = "data/train",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        super(DataProcessor, self).__init__()

        self.mean, self.std = _compute_mean_std(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

        full_dataset = ImageFolder(root=data_dir, transform=self.val_transform)
        self.class_names = full_dataset.classes

        total_size = len(full_dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        train_dataset_full = ImageFolder(root=data_dir, transform=self.train_transform)
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices

        train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        )

    def get_loaders(self):
        """Return training and validation data loaders"""
        return self.train_loader, self.val_loader

    def get_class_names(self):
        """Return the class names from the dataset"""
        return self.class_names
