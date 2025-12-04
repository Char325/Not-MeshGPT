# scripts/point_cloud_autoen.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# FoldingNet-style Encoder
# -----------------------------
class FoldingNetEncoder(nn.Module):
    """
    Deeper PointNet-style encoder with Conv1d + BatchNorm.
    Input:  (B, N, 3)
    Output: (B, latent_dim)
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, latent_dim, 1)
        self.bn4 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))   # (B, latent_dim, N)

        # global max pool over points
        x = torch.max(x, dim=2)[0]    # (B, latent_dim)
        return x


class FoldingNetDecoder(nn.Module):
    def __init__(self, num_points=2048, latent_dim=256):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        grid_size = int(math.sqrt(num_points))
        self.grid_size = grid_size

        # Stage 1 folding
        self.fold1 = nn.Sequential(
            nn.Linear(latent_dim + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        # Stage 2 folding (concat output of stage1 with latent)
        self.fold2 = nn.Sequential(
            nn.Linear(latent_dim + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        # Precomputed grid
        lin = torch.linspace(-1, 1, grid_size)
        X, Y = torch.meshgrid(lin, lin, indexing="xy")
        grid = torch.stack([X, Y], dim=-1).view(-1, 2)
        self.register_buffer("grid", grid)

    def forward(self, z):
        B = z.size(0)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)  # (B, P, 2)

        # Concatenate latent to 2D grid
        z_expanded = z.unsqueeze(1).expand(-1, grid.size(1), -1)

        inp1 = torch.cat([grid, z_expanded], dim=2)
        fold1 = self.fold1(inp1)

        # Stage 2
        inp2 = torch.cat([fold1, z_expanded], dim=2)
        fold2 = self.fold2(inp2)

        return fold2[:, :self.num_points, :]



# -----------------------------
# FoldingNet Autoencoder
# -----------------------------
class FoldingNetAE(nn.Module):
    def __init__(self, num_points=2048, latent_dim=256):
        super().__init__()
        self.encoder = FoldingNetEncoder(latent_dim=latent_dim)
        self.decoder = FoldingNetDecoder(num_points=num_points, latent_dim=latent_dim)

    def forward(self, x):
        """
        x: (B, N, 3)
        returns:
            recon: (B, N, 3)
            z:     (B, latent_dim)
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def chamfer_distance(pc1, pc2):
    # pc1, pc2: (B, N, 3)
    x, y = pc1, pc2
    x2 = torch.sum(x ** 2, dim=2, keepdim=True)  # (B,N,1)
    y2 = torch.sum(y ** 2, dim=2, keepdim=True)  # (B,N,1)
    xy = torch.bmm(x, y.transpose(1, 2))         # (B,N,N)

    dist = x2 + y2.transpose(1, 2) - 2 * xy

    min1 = torch.min(dist, dim=2)[0]
    min2 = torch.min(dist, dim=1)[0]

    return min1.mean() + min2.mean()



# backwards-compatible alias so existing imports still work
PointNetAE = FoldingNetAE
