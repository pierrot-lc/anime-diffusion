"""Implementation of U-ViT model.

For more information, see https://arxiv.org/pdf/2209.12152.pdf.
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from positional_encodings.torch_encodings import PositionalEncoding1D

from .backbone import Backbone


class UViT(nn.Module):
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
        assert (
            image_size % n_tokens == 0
        ), "The image size is not a multiple of n_tokens."
        kernel_size = image_size // n_tokens

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                stride=kernel_size,
            ),
            nn.GELU(),
            nn.LayerNorm([hidden_size, n_tokens, n_tokens]),
            Rearrange("b c h w -> b (h w) c"),
        )

        self.positional_encoding = PositionalEncoding1D(hidden_size)

        self.backbone = Backbone(hidden_size, n_heads, ff_size, dropout, n_layers)

        self.project = nn.Sequential(
            nn.Linear(hidden_size, n_channels * kernel_size * kernel_size),
            Rearrange(
                "b (h w) (c k1 k2) -> b c (h k1) (w k2)",
                h=n_tokens,
                w=n_tokens,
                k1=kernel_size,
                k2=kernel_size,
            ),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding="same"),
        )

    def forward(self, images: torch.Tensor, conditionals: torch.Tensor) -> torch.Tensor:
        """U-ViT forward pass, encoding the input images.

        ---
        Args:
            images: Input images.
                Shape of [batch_size, n_channels, image_size, image_size].
            conditionals: Input conditionals.
                Shape of [batch_size, n_conditionals, hidden_size].

        ---
        Returns:
            x: Predicted logits.
                Shape of [batch_size, n_channels, image_size, image_size].
        """
        n_conditionals = conditionals.shape[1]

        # Create the input tokens.
        patches = self.patch_embedding(images)
        tokens = torch.cat([conditionals, patches], dim=1)
        tokens = self.positional_encoding(tokens) + tokens

        # Main U-ViT backbone.
        tokens = self.backbone(tokens)

        # Remove the conditionals.
        tokens = tokens[:, n_conditionals:]

        # Predict the logits.
        logits = self.project(tokens)
        return logits
