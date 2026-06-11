import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from src.data.frame_dataset import FrameDataset
from src.models.autoencoder import ConvAutoencoder


FRAME_DIR = Path("data/frames")
OUTPUT_DIR = Path("results/shared/autoencoder")
LATENT_DIM = 128
BATCH_SIZE = 128
EPOCHS_PER_SESSION = 20
NUM_SESSIONS = 2
LEARNING_RATE = 1e-4
SEED = 42
DEVICE = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the convolutional autoencoder.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing checkpoint.",
    )
    return parser.parse_args()


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    train_losses,
    val_losses,
    train_indices,
    val_indices,
    completed_sessions,
):
    config = {
        "frame_dir": str(FRAME_DIR),
        "latent_dim": LATENT_DIM,
        "batch_size": BATCH_SIZE,
        "epochs_per_session": EPOCHS_PER_SESSION,
        "num_sessions": NUM_SESSIONS,
        "total_epochs": EPOCHS_PER_SESSION * NUM_SESSIONS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "optimizer_reinitialized_between_sessions": True,
        "device_requested": DEVICE,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latent_dim": LATENT_DIM,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "completed_sessions": completed_sessions,
            "config": config,
        },
        checkpoint_path,
    )
    with (OUTPUT_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                **config,
                "checkpoint": str(checkpoint_path),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "final_train_loss": train_losses[-1],
                "final_validation_loss": val_losses[-1],
            },
            file,
            indent=2,
        )


def main(overwrite=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"autoencoder_latent{LATENT_DIM}.pt"
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists: {checkpoint_path}. "
            "Use --overwrite only when intentionally retraining it."
        )

    set_reproducible_seed(SEED)
    dataset = FrameDataset(FRAME_DIR)
    split_generator = torch.Generator().manual_seed(SEED)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    shuffle_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []

    print(f"Dataset size: {len(dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Device: {device}")

    total_epochs = EPOCHS_PER_SESSION * NUM_SESSIONS
    optimizer = None
    for session in range(NUM_SESSIONS):
        # The historical experiment used two 20-epoch sessions and recreated
        # Adam before the second session while retaining the model weights.
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        for session_epoch in range(EPOCHS_PER_SESSION):
            epoch = session * EPOCHS_PER_SESSION + session_epoch + 1
            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )
            val_loss = evaluate(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(
                f"Epoch {epoch}/{total_epochs} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f}"
            )

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            train_losses,
            val_losses,
            list(train_dataset.indices),
            list(val_dataset.indices),
            completed_sessions=session + 1,
        )

    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    args = parse_args()
    main(overwrite=args.overwrite)
