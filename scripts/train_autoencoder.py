import torch
from pathlib import Path
from src.data.frame_dataset import FrameDataset
from src.models.autoencoder import ConvAutoencoder
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim import Adam

FRAME_DIR = Path("data/frames")
CHECKPOINT_DIR = Path("results/checkpoints/autoencoder")
PLOT_DIR = Path("results/plots/autoencoder")
LATENT_DIM = 128
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-4
GENERATOR_SEED = 42
DEVICE = "cuda"
CONTINUE_TRAINING = False


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        reconstructions = model(images)
        loss = criterion(reconstructions, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(train_loader.dataset)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            reconstructions = model(images)
            loss = criterion(reconstructions, targets)

            running_loss += loss.item() * images.size(0)

    return running_loss / len(val_loader.dataset)
    

def main():

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = FrameDataset(FRAME_DIR)

    generator = torch.Generator().manual_seed(GENERATOR_SEED)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    checkpoint_path = CHECKPOINT_DIR / f"autoencoder_latent{LATENT_DIM}.pt"

    if CONTINUE_TRAINING:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot continue training: {checkpoint_path} not found")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]

    print(f"Dataset size: {len(dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Device: {device}")

    start_epoch = len(train_losses)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {start_epoch + epoch + 1}/{start_epoch + EPOCHS} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": LATENT_DIM,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        checkpoint_path
    )

if __name__ == "__main__":
    main()
