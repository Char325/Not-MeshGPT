# train_transformer_ae.py

import os
import torch
from torch.utils.data import DataLoader
from torch import optim

from scripts.pytorch_ds import ModelNet10PC
from scripts.transformer_folding_ae import (
    TransformerFoldingAE,
    chamfer_distance,
    repulsion_loss,
    smoothness_loss,
)


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

    dataset = ModelNet10PC(data_root)
    print("Total samples:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=2,        # transformers are memory-hungry
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    num_points = 2048
    d_model = 128
    latent_dim = 128

    model = TransformerFoldingAE(
        num_points=num_points,
        d_model=d_model,
        latent_dim=latent_dim,
        num_layers=4,
        nhead=4,
        dim_feedforward=256,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # loss weights
    w_chamfer = 1.0
    w_repulse = 0.1
    w_smooth = 0.05

    num_epochs = 100

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)           # (B,N,3)

            optimizer.zero_grad()
            recon, _ = model(batch)

            loss_chamfer = chamfer_distance(batch, recon)
            loss_repulse = repulsion_loss(recon)
            loss_smooth = smoothness_loss(recon)

            loss = (
                w_chamfer * loss_chamfer
                + w_repulse * loss_repulse
                + w_smooth * loss_smooth
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 20 == 0:
                print(
                    f"Epoch {epoch} | Step {i+1}/{len(dataloader)} | "
                    f"Loss: {loss.item():.6f} "
                    f"(Chamfer={loss_chamfer.item():.6f}, "
                    f"Repulse={loss_repulse.item():.6f}, "
                    f"Smooth={loss_smooth.item():.6f})"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch}/{num_epochs} | Avg Loss: {avg_loss:.6f}")

        os.makedirs("checkpoints_transformer_pp", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"checkpoints_transformer_pp/transformer_foldingpp_epoch{epoch}.pth",
        )


if __name__ == "__main__":
    main()
