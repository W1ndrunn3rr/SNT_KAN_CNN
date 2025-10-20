from src.models.snt_cnn import SNTCNN
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import ssl
import hydra
from omegaconf import DictConfig

ssl._create_default_https_context = ssl._create_unverified_context
torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def test(cfg: DictConfig):
    """
    Test a trained model on the test dataset.

    Args:
        cfg: Hydra configuration containing:
            - test_params.checkpoint_path: Path to the model checkpoint
            - data_params.test_dir: Path to test data directory
            - data_params.batch_size: Batch size for testing
            - data_params.num_workers: Number of data loading workers
    """
    torch.backends.cudnn.benchmark = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    checkpoint_path = os.path.join(project_root, cfg.test_params.checkpoint_path)
    test_dir = os.path.join(project_root, cfg.data_params.test_dir)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please check if the path is correct in config.yaml or override with:\n"
            f"test_params.checkpoint_path=path/to/your/checkpoint.ckpt"
        )

    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            f"Please check if the path is correct in config.yaml"
        )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data_params.batch_size,
        shuffle=False,
        num_workers=cfg.data_params.num_workers,
        pin_memory=True,
    )

    class_names = test_dataset.classes

    print(f"🔄 Loading model from checkpoint:")
    print(f"   {checkpoint_path}")

    model = SNTCNN.load_from_checkpoint(
        checkpoint_path,
        class_names=class_names,
    )

    print(f"Model loaded successfully!")
    print(f"Model type: {model.model_type}")
    print(f"Number of classes: {model.num_classes}")

    logs_dir = os.path.join(project_root, "logs")
    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="tensorboard",
        version=f"{model.model_type}_test",
        default_hp_metric=False,
    )

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=cfg.data_params.precision,
        logger=tb_logger,
        enable_model_summary=True,
    )

    trainer.test(model, test_loader)


if __name__ == "__main__":
    test()
