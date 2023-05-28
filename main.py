from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.dataset import ImageDataset
from src.model.diffusion import DiffusionModel


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
    model.summary(config.device)
    return model


def init_dataset(config: DictConfig) -> ImageDataset:
    dataset = ImageDataset.from_folder(config.data.path, config.data.image_size)
    return dataset


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig):
    config.data.path = Path(to_absolute_path(config.data.path))
    if config.device == "auto":
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    model = init_model(config)
    dataset = init_dataset(config)


if __name__ == "__main__":
    # Launch with hydra.
    main()
