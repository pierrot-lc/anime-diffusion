from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import wandb
from einops import rearrange
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad
from tqdm import tqdm

from .model import DiffusionModel


class Trainer:
    def __init__(
        self,
        model: DiffusionModel,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: Optimizer,
        n_steps: int,
        n_epochs: int,
        device: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.device = device

        self.betas = torch.linspace(0.0001, 0.02, self.n_steps, device=self.device)
        self.loss_fn = nn.HuberLoss(reduction="mean")

    @torch.no_grad()
    def sample_images(self, images: torch.Tensor) -> torch.Tensor:
        """Use the model to sample using the given initial noise."""
        for timestep in reversed(range(self.n_steps)):
            timesteps = torch.full(
                (images.shape[0],), fill_value=timestep, device=self.device
            )

            # Subtract the mean noise.
            factor = (
                self.one_minus_alphas[timesteps]
                / self.sqrt_one_minus_alphas_cumprod[timesteps]
            )
            factor = rearrange(factor, "b -> b () () ()")
            images = images - factor * self.model(images, timesteps)

            factor = self.sqrt_alphas[timesteps]
            factor = rearrange(factor, "b -> b () () ()")
            images = images / factor

            # Add random variance.
            if timestep != 0:
                z = torch.randn_like(images, device=self.device)
            else:
                z = torch.zeros_like(images, device=self.device)
            factor = self.posterior_variance[timesteps]
            factor = rearrange(factor, "b -> b () () ()")
            images = images + factor * z

        images = (images + 1) / 2  # To range [0, 1].
        images.clip_(0, 1)
        return images

    def training_step(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Do a training step over the given images and compute metrics.

        ---
        Args:
            images: The images to train on.
                Shape of [batch_size, n_channels, image_size, image_size].

        ---
        Returns:
            A dictionary containing the metrics.
        """
        # Sample timesteps and noises.
        timesteps = torch.randint(
            0, self.n_steps, (images.shape[0],), device=self.device
        )
        noises = torch.randn_like(images, device=self.device)

        # Compute the noisy images.
        blur_factor = rearrange(self.sqrt_alphas_cumprod[timesteps], "b -> b () () ()")
        noise_factor = rearrange(
            self.sqrt_one_minus_alphas_cumprod[timesteps], "b -> b () () ()"
        )
        noisy_images = blur_factor * images + noise_factor * noises

        # Predict the noises.
        predicted_noises = self.model(noisy_images, timesteps)

        # Compute the loss.
        loss = self.loss_fn(noises, predicted_noises)

        return {"loss": loss}

    def do_epoch(self):
        """Do a training epoch."""
        self.model.train()
        for images in tqdm(self.train_loader, desc="Training step", leave=False):
            images = images.to(self.device)
            metrics = self.training_step(images)

            self.optimizer.zero_grad()
            metrics["loss"].backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, torch.Tensor]:
        self.model.eval()
        all_metrics = defaultdict(list)
        for images in tqdm(loader, desc="Eval step", leave=False):
            images = images.to(self.device)
            metrics = self.training_step(images)
            for key, value in metrics.items():
                all_metrics[key].append(value)

        return {key: torch.stack(value).mean() for key, value in all_metrics.items()}

    def launch_training(self, config: dict[str, Any], mode: str):
        assert mode in ["online", "offline", "disabled"]

        self.model.to(self.device)

        with wandb.init(
            project="anime-diffusion", entity="pierrotlc", config=config, mode=mode
        ) as run:
            if mode != "disabled":
                self.model.summary(self.device)

            for _ in tqdm(range(self.n_epochs), desc="Epoch"):
                self.do_epoch()

                metrics = self.evaluate(self.test_loader)
                images = torch.randn(
                    6,
                    self.model.n_channels,
                    self.model.image_size,
                    self.model.image_size,
                    device=self.device,
                )
                images = self.sample_images(images)
                metrics["images"] = wandb.Image(images)
                run.log(metrics)

    @property
    def alphas(self) -> torch.Tensor:
        return 1.0 - self.betas

    @property
    def sqrt_alphas(self) -> torch.Tensor:
        return torch.sqrt(self.alphas)

    @property
    def one_minus_alphas(self) -> torch.Tensor:
        return 1.0 - self.alphas

    @property
    def alphas_cumprod(self) -> torch.Tensor:
        return torch.cumprod(self.alphas, dim=0)

    @property
    def sqrt_alphas_cumprod(self) -> torch.Tensor:
        return torch.sqrt(self.alphas_cumprod)

    @property
    def sqrt_one_minus_alphas_cumprod(self) -> torch.Tensor:
        return torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def posterior_variance(self) -> torch.Tensor:
        alphas_cumprod_prev = torch.concat(
            [torch.FloatTensor([1.0]).to(self.device), self.alphas_cumprod[:-1]], dim=0
        )
        return self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
