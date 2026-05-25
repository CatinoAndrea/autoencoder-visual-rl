from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    """Dataset of RGB CarRacing frames for autoencoder training.

    Each sample is returned as (image, image), because an autoencoder learns to
    reconstruct its own input.
    """

    def __init__(self, frame_dir: str | Path):
        self.frame_dir = Path(frame_dir)
        self.frame_paths = sorted(self.frame_dir.glob("*.png"))

        if not self.frame_paths:
            raise FileNotFoundError(f"No PNG frames found in {self.frame_dir}")

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int):
        frame_path = self.frame_paths[index]

        image = Image.open(frame_path).convert("RGB")

        # Convert uint8 pixels from [0, 255] to float32 values in [0, 1].
        image = np.asarray(image, dtype=np.float32) / 255.0

        # Convert from H x W x C to C x H x W for PyTorch.
        image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image)

        # For an autoencoder, the target is the original image itself.
        return image, image

