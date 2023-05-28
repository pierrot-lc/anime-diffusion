from src.model.diffusion import DiffusionModel

model = DiffusionModel(
    image_size=256,
    n_channels=3,
    n_tokens=8,
    hidden_size=256,
    n_heads=4,
    ff_size=1024,
    dropout=0.1,
    n_layers=6,
)

model.summary("cpu")
