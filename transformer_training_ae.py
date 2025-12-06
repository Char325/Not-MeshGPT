import os
import torch
from torch.utils.data import DataLoader
from scripts.pytorch_ds import ModelNet10PC
from scripts.point_cloud_autoen2 import PointNetAE, chamfer_distance
from torch.optim import Adam


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():

    data_root = "data/modelnet10_pc_2048"
    device = get_device()
    print("Using device:", device)

    # Build dataset: exact 70/10/20 split
    train_ds = ModelNet10PC(data_root, split="train")
    val_ds   = ModelNet10PC(data_root, split="val")

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # AE
    model = PointNetAE(num_points=2048, latent_dim=256).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epochs = 80

    for epoch in range(1, num_epochs + 1):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0

        for pc in train_loader:
            pc = pc.to(device)
            optimizer.zero_grad()

            recon, _ = model(pc)
            loss = chamfer_distance(pc, recon)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for pc in val_loader:
                pc = pc.to(device)
                recon, _ = model(pc)
                val_loss += chamfer_distance(pc, recon).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        os.makedirs("checkpoints_transformer_pp", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints_transformer_pp/transformer_foldingpp_epoch{epoch}.pth")


if __name__ == "__main__":
    main()
