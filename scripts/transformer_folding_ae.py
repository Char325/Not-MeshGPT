# scripts/transformer_folding_ae.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Transformer Encoder Blocks for Point Clouds
# -------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, B, C)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ff = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout(ff)
        x = self.norm2(x)
        return x


class PointTransformerEncoder(nn.Module):
    """
    Simple transformer encoder for point clouds.
    Input:  (B, N, 3)
    Output: (B, latent_dim)
    """
    def __init__(self, d_model=128, num_layers=4, nhead=4,
                 dim_feedforward=256, latent_dim=128):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(3, d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.0
            ) for _ in range(num_layers)
        ])

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.latent_linear = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        x = self.input_proj(x)          # (B, N, d_model)

        x = x.transpose(0, 1)           # (N, B, d_model)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1).transpose(1, 2)  # (B, d_model, N)

        x = self.pool(x).squeeze(-1)    # (B, d_model)
        z = self.latent_linear(x)       # (B, latent_dim)
        return z


# -------------------------------------------------
# FoldingNet++ Decoder (2-stage + learned offsets)
# -------------------------------------------------
class FoldingNetPPDecoder(nn.Module):
    """
    FoldingNet++ style decoder:
    - 2D grid + learned offsets
    - two folding stages
    - residual refinement
    """
    def __init__(self, num_points=2048, latent_dim=128):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        grid_size = int(math.sqrt(num_points))
        self.grid_size = grid_size

        # base grid
        lin = torch.linspace(-1, 1, grid_size)
        X, Y = torch.meshgrid(lin, lin, indexing="xy")
        grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # (P,2)
        self.register_buffer("grid", grid)

        P = grid.shape[0]

        # learned 2D offsets per grid point
        self.offset_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        # Stage 1 folding
        self.fold1 = nn.Sequential(
            nn.Linear(latent_dim + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        # Stage 2 folding (refinement with residual)
        self.fold2 = nn.Sequential(
            nn.Linear(latent_dim + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

        # tiny residual tweak after second fold
        self.residual = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, z):
        """
        z: (B, latent_dim)
        returns: (B, num_points, 3)
        """
        B = z.size(0)
        P = self.grid.size(0)

        # base grid & learned offsets
        grid = self.grid.unsqueeze(0).expand(B, -1, -1)      # (B,P,2)
        offset = self.offset_mlp(grid)                       # (B,P,2)
        grid = grid + offset                                 # (B,P,2)

        # broadcast latent to each point
        z_expanded = z.unsqueeze(1).expand(-1, P, -1)        # (B,P,latent)

        # Stage 1
        inp1 = torch.cat([grid, z_expanded], dim=2)          # (B,P,latent+2)
        fold1 = self.fold1(inp1)                             # (B,P,3)

        # Stage 2
        inp2 = torch.cat([fold1, z_expanded], dim=2)         # (B,P,latent+3)
        fold2 = self.fold2(inp2)                             # (B,P,3)

        # residual refinement
        res = self.residual(fold2)
        out = fold2 + 0.1 * res                              # (B,P,3)

        return out[:, :self.num_points, :]


# -------------------------------------------------
# Full Transformer + FoldingNet++ Autoencoder
# -------------------------------------------------
class TransformerFoldingAE(nn.Module):
    def __init__(self, num_points=2048, d_model=128, latent_dim=128,
                 num_layers=4, nhead=4, dim_feedforward=256):
        super().__init__()
        self.encoder = PointTransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            latent_dim=latent_dim
        )
        self.decoder = FoldingNetPPDecoder(
            num_points=num_points,
            latent_dim=latent_dim
        )

    def forward(self, x):
        # x: (B, N, 3)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# -------------------------------------------------
# Chamfer + Repulsion + Smoothness losses
# -------------------------------------------------
def chamfer_distance(pc1, pc2):
    """
    pc1, pc2: (B, N, 3)
    """
    x, y = pc1, pc2
    x2 = torch.sum(x ** 2, dim=2, keepdim=True)       # (B,N,1)
    y2 = torch.sum(y ** 2, dim=2, keepdim=True)       # (B,N,1)
    xy = torch.bmm(x, y.transpose(1, 2))              # (B,N,N)

    dist = x2 + y2.transpose(1, 2) - 2 * xy           # (B,N,N)

    min1 = torch.min(dist, dim=2)[0]                  # (B,N)
    min2 = torch.min(dist, dim=1)[0]                  # (B,N)

    return min1.mean() + min2.mean()


def repulsion_loss(pc, k=10, h=0.03):
    """
    Encourage points to spread out (no clustering).
    pc: (B, N, 3)
    """
    B, N, _ = pc.shape
    with torch.no_grad():
        dist = torch.cdist(pc, pc, p=2) + 1e-8        # (B,N,N)
        knn_dist, _ = torch.topk(dist, k=k+1, dim=-1, largest=False)
        # ignore self-distance (0)
        knn_dist = knn_dist[:, :, 1:]                 # (B,N,k)

    loss = torch.exp(- (knn_dist ** 2) / (h ** 2)).mean()
    return loss


def smoothness_loss(pc, k=10):
    """
    Laplacian smoothness: each point close to mean of its neighbors.
    pc: (B, N, 3)
    """
    dist = torch.cdist(pc, pc, p=2) + 1e-8            # (B,N,N)
    knn_dist, knn_idx = torch.topk(dist, k=k+1, dim=-1, largest=False)
    knn_idx = knn_idx[:, :, 1:]                       # drop self index

    B, N, _ = pc.shape
    idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B,N,k,3)
    neighbors = torch.gather(
        pc.unsqueeze(1).expand(B, N, N, 3),  # (B,N,N,3)
        2,
        idx_expanded
    )                                        # (B,N,k,3)

    neighbor_mean = neighbors.mean(dim=2)    # (B,N,3)
    lap = pc - neighbor_mean                 # (B,N,3)

    return (lap ** 2).mean()
