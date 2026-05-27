import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mediapipe_adjacency(num_joints=33):
    """Return row-normalized MediaPipe-style skeleton adjacency."""
    A = torch.eye(num_joints)
    edges = [
        (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (11, 12), (23, 24), (11, 23), (12, 24),
    ]
    for i, j in edges:
        if i < num_joints and j < num_joints:
            A[i, j] = A[j, i] = 1
    degree = torch.sum(A, dim=1, keepdim=True) + 1e-6
    return A / degree


class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A=None, dropout=0.0):
        super().__init__()
        if A is not None:
            self.register_buffer("A", A)
        else:
            self.A = None
        self.theta = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A=None):
        # x: (B, T, J, C)
        x = self.theta(x)                              # (B, T, J, outC)
        graph = A if A is not None else self.A
        if graph is None:
            raise ValueError("GCNBlock requires an adjacency matrix.")
        graph = graph.to(x.device)
        x = torch.einsum("ij,btjc->btic", graph, x)    # graph conv
        x = self.dropout(x)
        return x


class ResGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.gcn = GCNBlock(in_c, out_c, A)
        self.proj = nn.Linear(in_c, out_c) if in_c != out_c else None
        self.norm = nn.LayerNorm(out_c)

    def forward(self, x):
        res = x if self.proj is None else self.proj(x)
        x = self.gcn(x)
        x = self.norm(x + res)
        return F.relu(x)


class JointAttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, T, J, C)
        w = torch.softmax(self.attn(x), dim=2)
        return (x * w).sum(dim=2)       # (B, T, C)


class TemporalTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        # x: (B, T, C)
        return self.encoder(x)


class HybridCOMGCNv2(nn.Module):
    """
    COM-anchored Hybrid GCN v2:
    - Residual GCN stacks with learnable adjacency
    - Joint attention pooling
    - Temporal Transformer
    - Regression head
    """

    def __init__(self, in_channels=9, num_joints=33, hidden=128):
        super().__init__()

        A = build_mediapipe_adjacency(num_joints)
        self.register_buffer("A", A)

        self.gcn1 = ResGCNBlock(in_channels, 64, self.A)
        self.gcn2 = ResGCNBlock(64, 128, self.A)
        self.gcn3 = ResGCNBlock(128, hidden, self.A)

        self.joint_pool = JointAttentionPool(hidden)

        self.temporal_tf = TemporalTransformer(
            dim=hidden,
            num_heads=4,
            ff_dim=hidden * 2
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, T, J, C)
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = self.joint_pool(x)     # (B, T, C)
        x = self.temporal_tf(x)    # (B, T, C)
        x = x.mean(dim=1)          # temporal pooling
        return self.regressor(x).squeeze(-1)


# Backward-compatible name used by older ablation scripts.
HybridCOMGCN = HybridCOMGCNv2
