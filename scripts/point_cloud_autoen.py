

# scripts/pointnet_ae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    Simple PointNet-style encoder:
    Input: (B, N, 3)
    Output: (B, latent_dim)
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, latent_dim)

    def forward(self, x):
        # x: (B, N, 3)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)          # (B, N, latent_dim)
        x = torch.max(x, dim=1)[0]  # global max pool over points -> (B, latent_dim)
        return x


class PointNetDecoder(nn.Module):
    """
    Simple MLP decoder:
    Input: (B, latent_dim)
    Output: (B, N, 3)
    """
    def __init__(self, num_points=2048, latent_dim=128):
        super().__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, num_points * 3)

    def forward(self, z):
        # z: (B, latent_dim)
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                     # (B, N*3)
        x = x.view(-1, self.num_points, 3)  # (B, N, 3)
        return x


class PointNetAE(nn.Module):
    def __init__(self, num_points=2048, latent_dim=128):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim=latent_dim)
        self.decoder = PointNetDecoder(num_points=num_points, latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def chamfer_distance(pcd1, pcd2):
    """
    pcd1, pcd2: (B, N, 3)
    Simple Chamfer distance using torch.cdist.
    NOTE: Use moderate batch sizes (e.g., 8) to avoid memory issues.
    """
    # pairwise distances: (B, N, N)
    dists = torch.cdist(pcd1, pcd2, p=2)

    # for each point in set1, find closest in set2
    min1, _ = torch.min(dists, dim=2)  # (B, N)
    # for each point in set2, find closest in set1
    min2, _ = torch.min(dists, dim=1)  # (B, N)

    loss = min1.mean() + min2.mean()
    return loss



