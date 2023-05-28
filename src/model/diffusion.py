"""Diffusion model."""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torchinfo import ModelStatistics, summary

from .time_encoding import TimeEncoding
from .uvit import UViT


class DiffusionModel(nn.Module):
    def __init__(
        self,
        image_size: int,
        n_channels: int,
        n_tokens: int,
        hidden_size: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        n_layers: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_channels = n_channels

        self.time_encoding = nn.Sequential(
            TimeEncoding(hidden_size),
            Rearrange("b h -> b () h"),  # Add the token dimension.
        )

        self.uvit = UViT(
            image_size,
            n_channels,
            n_tokens,
            hidden_size,
            n_heads,
            ff_size,
            dropout,
            n_layers,
        )

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Diffusion forward pass, predicting the noise in the images for the
        given timesteps.

        ---
        Args:
            images: Noisy images.
                Shape of [batch_size, n_channels, image_size, image_size].
            timesteps: Timesteps to predict the noise for.
                Shape of [batch_size, n_timesteps].

        ---
        Returns:
            Noise for the given timesteps.
                Shape of [batch_size, n_channels, image_size, image_size].
        """
        time_tokens = self.time_encoding(timesteps)
        noise = self.uvit(images, time_tokens)
        return noise

    def summary(self, device: str) -> ModelStatistics:
        images = torch.zeros(
            (1, self.n_channels, self.image_size, self.image_size),
            dtype=torch.float,
            device=device,
        )
        timesteps = torch.zeros(
            (1,),
            dtype=torch.long,
            device=device,
        )

        return summary(self, input_data=(images, timesteps), device=device)
