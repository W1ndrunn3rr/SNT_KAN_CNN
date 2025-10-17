from src.data_processing.data_processor import DataProcessor
from src.models.diat_kan import DIATKAN
import lightning as L
import os
import ssl
import hydra
import torch

ssl._create_default_https_context = ssl._create_unverified_context


@hydra.main(config_path="../config", config_name="config", version_base=None)
def train(cfg):
    torch.backends.cudnn.benchmark = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    models_dir = os.path.join(project_root, cfg.log_params.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    data_dir = os.path.join(project_root, cfg.data_params.train_dir)

    data_processor = DataProcessor()

    train_loader, val_loader = data_processor.get_loaders()

    model = DIATKAN(model_type=cfg.model_params.model_type)

    trainer = L.Trainer(
        max_epochs=cfg.model_params.max_epochs,
        accelerator="gpu",
        precision="16-mixed",
    )

    trainer.fit(model, train_loader, val_loader)

    checkpoint_filename = f"{cfg.model_params.model_name}.ckpt"
    trainer.save_checkpoint(os.path.join(models_dir, checkpoint_filename))


if __name__ == "__main__":
    train()
