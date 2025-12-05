# train_ae.py

import os
import torch
from torch.utils.data import DataLoader
from torch import optim

from scripts.pytorch_ds import ModelNet10PC
from scripts.point_cloud_autoen2 import PointNetAE, chamfer_distance


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    data_root = "data/modelnet10_pc_2048"  # adjust if needed

    device = get_device()
    print("Using device:", device)

    # Dataset & DataLoader
    dataset = ModelNet10PC(data_root)
    print("Total samples:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=4,      # keep small for Chamfer + cdist
        shuffle=True,
        num_workers=0,     # set >0 on Linux if you like
        drop_last=True
    )

    num_points = 2048
    latent_dim = 256

    model = PointNetAE(num_points=num_points, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 80

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for i, batch in enumerate(dataloader):
            # batch: (B, N, 3)
            batch = batch.to(device)

            optimizer.zero_grad()
            recon, z = model(batch)
            loss = chamfer_distance(batch, recon)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 20 == 0:
                print(f"Epoch {epoch} | Step {i+1}/{len(dataloader)} | Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.6f}")

        # save a checkpoint every few epochs
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/pointnet_ae_epoch{epoch}.pth")


if __name__ == "__main__":
    main()
