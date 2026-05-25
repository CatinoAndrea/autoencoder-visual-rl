import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """CNN autoencoder for 96x96 RGB CarRacing frames.

    The encoder compresses an image into a compact latent vector. The decoder
    reconstructs the original image from that latent representation.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Shape after encoder_cnn: 256 x 6 x 6.
        self.flatten_dim = 256 * 6 * 6

        self.encoder_fc = nn.Linear(self.flatten_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into latent vectors."""
        x = self.encoder_cnn(x)
        x = torch.flatten(x, start_dim=1)
        z = self.encoder_fc(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back into reconstructed images."""
        x = self.decoder_fc(z)

        # Convert flat vectors back into feature maps for the transposed CNN.
        x = x.view(-1, 256, 6, 6)
        x = self.decoder_cnn(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of input images."""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction
