import pytest
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D

from .time_encoding import TimeEncoding


@pytest.mark.parametrize(
    "hidden_size, timesteps",
    [
        (32, torch.tensor([0, 1, 2])),
        (64, torch.tensor([0, 4, 0, 15])),
        (128, torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])),
    ],
)
def test_time_encoding(hidden_size: int, timesteps: torch.Tensor):
    """Test the time encoding module."""
    time_encoding = TimeEncoding(hidden_size)
    position_enc = PositionalEncoding1D(hidden_size)

    # Test the shape of the output.
    encodings = time_encoding(timesteps)
    assert encodings.shape == torch.Size([len(timesteps), hidden_size])

    # Test the values of the output.
    max_timesteps = int(timesteps.max().item())
    x = torch.zeros(1, max_timesteps + 1, hidden_size, device=timesteps.device)
    true_encodings = position_enc(x).to(timesteps.device)
    true_encodings = true_encodings.squeeze(0)  # Shape is [timesteps, hidden_size].
    for step_id, step in enumerate(timesteps):
        assert torch.equal(true_encodings[step], encodings[step_id])
