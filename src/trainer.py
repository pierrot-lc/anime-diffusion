from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from torch.optim import Optimizer
from torch.utils.data import DataLoader
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
        noisy_images = self.q_sample(images, noises, timesteps)

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

                metrics: dict[str, Any] = self.evaluate(self.test_loader)
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

                image = next(iter(self.train_loader))[:1]
                image = image.to(self.device)
                self.q_gif(image[0], Path("q-sample.gif"))
                image = self.q_sample(
                    image, torch.randn_like(image), torch.LongTensor([self.n_steps - 1])
                )
                self.p_gif(image[0], Path("p-sample.gif"))

    @torch.no_grad()
    def q_sample(
        self, images: torch.Tensor, noises: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the diffusion distribution."""
        mean_factor = rearrange(self.sqrt_alphas_cumprod[timesteps], "b -> b () () ()")
        std_factor = rearrange(
            self.sqrt_one_minus_alphas_cumprod[timesteps], "b -> b () () ()"
        )
        noisy_images = mean_factor * images + std_factor * noises

        return noisy_images

    @torch.no_grad()
    def p_sample(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Sample from the reverse diffusion distribution."""
        # Subtract the mean noise.
        mean_factor = (
            self.betas[timesteps] / self.sqrt_one_minus_alphas_cumprod[timesteps]
        )
        mean_factor = rearrange(mean_factor, "b -> b () () ()")
        images = images - mean_factor * self.model(images, timesteps)

        mean_factor = self.sqrt_alphas[timesteps]
        mean_factor = rearrange(mean_factor, "b -> b () () ()")
        images = images / mean_factor

        # Add random variance.
        z = torch.randn_like(images, device=self.device)
        z[timesteps == 0] = 0
        std_factor = self.posterior_std[timesteps]
        std_factor = rearrange(std_factor, "b -> b () () ()")
        images = images + std_factor * z

        return images

    @torch.no_grad()
    def q_gif(self, image: torch.Tensor, gif_filename: Path):
        """Generate a GIF of the diffusion process.

        ---
        Args:
            image: The plain image (without noise) to start from.
                Shape of [n_channels, image_size, image_size].
            gif_filename: The path to the GIF file to save.
        """
        assert len(image.shape) == 3, "Expected a single image."

        gif = [image]
        noises = torch.randn((self.n_steps, *image.shape), device=self.device)

        image = image.unsqueeze(0)  # Add batch dimension.
        for t in range(self.n_steps):
            timestep = torch.full((1,), t, device=self.device, dtype=torch.long)
            noisy_image = self.q_sample(image, noises[t], timestep)
            gif.append(noisy_image[0])

        frames = torch.stack(gif)
        frames = Trainer.postprocess(frames)
        iio.imwrite(gif_filename, frames, duration=100)

    @torch.no_grad()
    def p_gif(self, image: torch.Tensor, gif_filename: Path):
        """Generate a GIF of the reverse diffusion process.

        ---
        Args:
            image: The plain noise to start from.
                Shape of [n_channels, image_size, image_size].
            gif_filename: The path to the GIF file to save.
        """
        assert len(image.shape) == 3, "Expected a single image."
        gif = [image]
        image = image.unsqueeze(0)

        for t in reversed(range(self.n_steps)):
            timestep = torch.full((1,), t, device=self.device, dtype=torch.long)
            image = self.p_sample(image, timestep)
            gif.append(image[0])

        frames = torch.stack(gif)
        frames = Trainer.postprocess(frames)
        iio.imwrite(gif_filename, frames, duration=100)

    @torch.no_grad()
    def sample_images(self, images: torch.Tensor) -> torch.Tensor:
        """Use the model to sample using the given initial noise."""
        for timestep in reversed(range(self.n_steps)):
            timesteps = torch.full(
                (images.shape[0],),
                fill_value=timestep,
                device=self.device,
                dtype=torch.long,
            )
            images = self.p_sample(images, timesteps)

        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images

    @property
    def alphas(self) -> torch.Tensor:
        return 1.0 - self.betas

    @property
    def sqrt_alphas(self) -> torch.Tensor:
        return torch.sqrt(self.alphas)

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

    @property
    def posterior_std(self) -> torch.Tensor:
        return torch.sqrt(self.posterior_variance)

    @staticmethod
    def postprocess(images: torch.Tensor) -> np.ndarray:
        """Denormalize the images and return a uint8 numpy array version."""
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images * 255
        images = rearrange(images, "b c h w -> b h w c")
        np_images = images.cpu().numpy()
        np_images = np_images.astype(np.uint8)
        return np_images
