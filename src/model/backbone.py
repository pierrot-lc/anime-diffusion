"""Implementation of the U-ViT.

It is a ViT with long skip connections.
For more details, see https://arxiv.org/pdf/2209.12152.pdf.
"""
import torch
import torch.nn as nn


class Backbone(nn.Module):
    """U-ViT backbone."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        ff_size: int,
        dropout: float,
        n_layers: int,
    ):
        """Initialize U-ViT backbone.

        ---
        Args:
            hidden_size: Size of the hidden states.
            n_heads: Number of attention heads.
            ff_size: Size of the feed-forward layer.
            dropout: Dropout probability.
            n_layers: Number of layers.
        """
        super().__init__()
        assert n_layers % 2 == 0, "Number of layers must be even."
        self.half_size = n_layers // 2

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_size, n_heads, ff_size, dropout, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_size * 2, hidden_size) for _ in range(self.half_size)]
        )
        self.final_block = nn.TransformerEncoderLayer(
            hidden_size, n_heads, ff_size, dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input sequence.

        ---
        Args:
            x: Sequence of patches.
                Shape of [batch_size, n_patches, hidden_size].

        ---
        Returns:
            x: Encoded sequence.
                Shape of [batch_size, n_patches, hidden_size].
        """
        residuals = []
        for block in self.blocks[: self.half_size]:
            x = block(x)
            residuals.append(x)

        for block, projection in zip(self.blocks[self.half_size :], self.projections):
            x = block(x)
            x = torch.cat([x, residuals.pop()], dim=-1)
            x = projection(x)

        x = self.final_block(x)

        return x
