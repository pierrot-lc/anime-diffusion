"""A simple time encoding module.
"""
import torch
import torch.nn as nn
from einops import repeat
from positional_encodings.torch_encodings import PositionalEncoding1D


class TimeEncoding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.position_enc = PositionalEncoding1D(hidden_size)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed the timesteps.

        ---
        Args:
            timesteps: The timesteps of the game states.
                Shape of [batch_size,].
        ---
        Returns:
            The positional encodings for the given timesteps.
                Shape of [batch_size, hidden_size].
        """
        # Compute the positional encodings for the given timesteps.
        max_timesteps = int(timesteps.max().item())
        x = torch.zeros(1, max_timesteps + 1, self.hidden_size, device=timesteps.device)
        encodings = self.position_enc(x).to(timesteps.device)
        encodings = encodings.squeeze(0)  # Shape is [timesteps, hidden_size].

        # Select the right encodings for the timesteps.
        encodings = repeat(encodings, "t e -> b t e", b=timesteps.shape[0])
        timesteps = repeat(timesteps, "b -> b t e", t=1, e=self.hidden_size)
        encodings = torch.gather(encodings, dim=1, index=timesteps)

        encodings = encodings.squeeze(1)  # Shape is [batch_size, hidden_size].
        return encodings
