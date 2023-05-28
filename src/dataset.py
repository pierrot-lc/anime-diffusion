from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import resize


class ImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], image_size: int):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Read the image at the given index.
        The image is normalized to the range [0, 1].

        ---
        Args:
            index: The index of the image to read.

        ---
        Returns:
            The image at the given index.
                Shape of [n_channels, images_size, images_size].
        """
        image_path = self.image_paths[index]
        image = read_image(str(image_path), mode=ImageReadMode.RGB)
        image = (image * 2 - 255) / 255.0  # Normalize to [-1, 1].
        image = resize(image, size=[self.image_size, self.image_size], antialias=True)
        return image

    @classmethod
    def from_folder(cls, folder_path: Path, image_size: int) -> "ImageDataset":
        image_paths = [path for path in folder_path.iterdir() if path.is_file()]
        return cls(image_paths, image_size)
