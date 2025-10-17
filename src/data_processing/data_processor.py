import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


class DataProcessor(nn.Module):
    def __init__(
        self, data_dir: str = "data", batch_size: int = 32, num_workers: int = 4
    ):
        super(DataProcessor, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.dataset = ImageFolder(root=data_dir, transform=self.transform)

        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def get_loaders(self):
        return self.train_loader, self.val_loader

    def get_class_names(self):
        """Return the class names from the dataset"""
        return self.dataset.classes
