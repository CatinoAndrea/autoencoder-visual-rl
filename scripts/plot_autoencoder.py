from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.frame_dataset import FrameDataset
from src.models.autoencoder import ConvAutoencoder


FRAME_DIR = Path("data/frames")
CHECKPOINT_PATH = Path("results/shared/autoencoder/autoencoder_latent128.pt")
PLOT_DIR = Path("results/shared/autoencoder/plots")
DEVICE = "cuda"
NUM_RECONSTRUCTIONS = 8


def tensor_to_image(image: torch.Tensor):
    """Convert a PyTorch image tensor from C x H x W to H x W x C."""
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    return image.clip(0.0, 1.0)


def plot_loss_curve(train_losses, val_losses, output_path: Path):
    """Save the autoencoder train/validation reconstruction loss curve."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE reconstruction loss")
    plt.title("Autoencoder training curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_reconstructions(model, dataset, device, output_path: Path):
    """Save a grid comparing original frames and reconstructed frames."""
    model.eval()

    indices = torch.linspace(0, len(dataset) - 1, steps=NUM_RECONSTRUCTIONS).long()
    images = torch.stack([dataset[int(index)][0] for index in indices]).to(device)

    with torch.no_grad():
        reconstructions = model(images)

    fig, axes = plt.subplots(2, NUM_RECONSTRUCTIONS, figsize=(2 * NUM_RECONSTRUCTIONS, 4))

    for col in range(NUM_RECONSTRUCTIONS):
        axes[0, col].imshow(tensor_to_image(images[col]))
        axes[0, col].axis("off")
        axes[0, col].set_title("Original", fontsize=9)

        axes[1, col].imshow(tensor_to_image(reconstructions[col]))
        axes[1, col].axis("off")
        axes[1, col].set_title("Reconstructed", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    model = ConvAutoencoder(latent_dim=checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = FrameDataset(FRAME_DIR)

    loss_curve_path = PLOT_DIR / "loss_curve.png"
    reconstructions_path = PLOT_DIR / "reconstructions.png"

    plot_loss_curve(
        checkpoint["train_losses"],
        checkpoint["val_losses"],
        loss_curve_path,
    )
    plot_reconstructions(model, dataset, device, reconstructions_path)

    print(f"Saved loss curve to {loss_curve_path}")
    print(f"Saved reconstructions to {reconstructions_path}")


if __name__ == "__main__":
    main()
