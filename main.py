from pathlib import Path

import hydra
import torch
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split

from src.dataset import ImageDataset
from src.model import DiffusionModel
from src.trainer import Trainer


def init_model(config: DictConfig) -> DiffusionModel:
    model = DiffusionModel(
        image_size=config.data.image_size,
        n_channels=config.data.n_channels,
        n_tokens=config.model.n_tokens,
        hidden_size=config.model.hidden_size,
        n_heads=config.model.n_heads,
        ff_size=config.model.ff_size,
        dropout=config.model.dropout,
        n_layers=config.model.n_layers,
    )
    return model


def init_datasets(config: DictConfig) -> tuple[Dataset, Dataset]:
    dataset = ImageDataset.from_folder(config.data.path, config.data.image_size)
    train_dataset, test_dataset = random_split(
        dataset,
        lengths=[0.8, 0.2],
        generator=torch.Generator().manual_seed(config.seed),
    )
    return train_dataset, test_dataset


def init_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = init_datasets(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
        shuffle=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        generator=torch.Generator().manual_seed(config.seed),
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, test_loader


def init_optimizer(config: DictConfig, model: DiffusionModel) -> optim.Optimizer:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    return optimizer


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig):
    config.data.path = Path(to_absolute_path(config.data.path))
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = init_model(config)
    train_loader, test_loader = init_dataloaders(config)
    optimizer = init_optimizer(config, model)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        n_steps=config.train.n_steps,
        n_epochs=config.train.n_epochs,
        device=config.device,
    )
    trainer.launch_training(OmegaConf.to_container(config), mode=config.mode)


if __name__ == "__main__":
    # Launch with hydra.
    main()
