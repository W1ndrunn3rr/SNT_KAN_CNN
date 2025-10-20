from src.data_processing.data_processor import DataProcessor
from src.models.snt_cnn import SNTCNN
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import ssl
import hydra
import torch

ssl._create_default_https_context = ssl._create_unverified_context
torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def train(cfg):
    torch.backends.cudnn.benchmark = True

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    logs_dir = os.path.join(project_root, "logs")
    models_dir = os.path.join(logs_dir, cfg.log_params.models_dir)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    tb_logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="tensorboard",
        version=f"{cfg.model_params.model_type}",
        default_hp_metric=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=models_dir,
        filename=f"{cfg.model_params.model_type}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    data_processor = DataProcessor()
    train_loader, val_loader = data_processor.get_loaders()
    class_names = data_processor.get_class_names()

    model = SNTCNN(
        model_type=cfg.model_params.model_type,
        num_classes=cfg.model_params.num_classes,
        optimizer=cfg.model_params.optimizer,
        learning_rate=cfg.model_params.learning_rate,
        class_names=class_names,
    )

    trainer = L.Trainer(
        max_epochs=cfg.model_params.max_epochs,
        accelerator="gpu",
        precision=cfg.data_params.precision,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_model_summary=False,
        log_every_n_steps=10,
        accumulate_grad_batches=cfg.data_params.gradient_accumulation_steps,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
