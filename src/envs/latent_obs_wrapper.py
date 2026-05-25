from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.models.autoencoder import ConvAutoencoder


class LatentObservationWrapper(gym.ObservationWrapper):
    """Replace RGB CarRacing frames with frozen autoencoder latent vectors."""

    def __init__(self, env, checkpoint_path: str | Path, device: str = "cpu"):
        super().__init__(env)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.latent_dim = checkpoint["latent_dim"]

        self.model = ConvAutoencoder(latent_dim=self.latent_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.latent_dim,),
            dtype=np.float32,
        )

    def observation(self, obs):
        # Match the preprocessing used during autoencoder training.
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (2, 0, 1))
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent = self.model.encode(obs_tensor)

        return latent.squeeze(0).cpu().numpy().astype(np.float32)
