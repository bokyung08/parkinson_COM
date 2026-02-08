import torch
import torch.nn as nn
import torch.nn.functional as F
from .hybrid_gcn import build_mediapipe_adjacency


class TransformerBlock(nn.Module):
    """Lightweight temporal Transformer encoder for gait dynamics."""

    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x: (B, T, C)
        return self.encoder(x)


class SpatialAttentionBlock(nn.Module):
    """
    Per-frame spatial self-attention + FFN, mirroring Keras SpatialTransformerBlock.
    Input: (B, T, J, C_in) -> Output: (B, T, J, d_model)
    """

    def __init__(self, in_channels=9, d_model=128, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, J, C_in) -> treat T frames independently
        B, T, J, C = x.shape
        x = self.in_proj(x)  # (B, T, J, d_model)
        x_reshaped = x.view(B * T, J, -1)  # (B*T, J, d_model)
        attn_out, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)
        x_reshaped = self.norm1(x_reshaped + self.dropout(attn_out))
        ffn_out = self.ffn(x_reshaped)
        x_reshaped = self.norm2(x_reshaped + self.dropout(ffn_out))
        return x_reshaped.view(B, T, J, -1)


class HybridFusionModel(nn.Module):
    """
    Final fusion model: COM-anchored GCN branch + spatial-smoothed temporal Transformer branch.
    Both branches share the same 9D node features and skeleton; outputs are fused for regression.
    """

    def __init__(self, in_channels=9, d_model=128, dropout=0.1):
        super().__init__()
        self.A = build_mediapipe_adjacency()

        # Branch 1: GCN (from HybridCOMGCN)
        from .hybrid_gcn import GCNBlock  # lazy import to avoid circular

        self.gcn1 = GCNBlock(in_channels, 64, dropout=dropout)
        self.gcn2 = GCNBlock(64, d_model, dropout=dropout)
        self.temporal_pool_gcn = nn.AdaptiveAvgPool1d(1)

        # Branch 2: spatial self-attention (per-frame) + temporal transformer (main-model spirit)
        self.spatial_attn = SpatialAttentionBlock(in_channels=in_channels, d_model=d_model, nhead=4, dim_feedforward=256, dropout=dropout)
        self.temporal_transformer = TransformerBlock(d_model=d_model, nhead=4, dim_feedforward=256, dropout=dropout)

        # Fusion head
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        """
        x: (B, T, J, C) where C=9 (ablation-aware if fewer channels are passed).
        """
        A = self.A.to(x.device)

        # ----- Branch 1: GCN -----
        g1 = self.gcn1(x, A)
        g2 = self.gcn2(g1, A)  # (B, T, J, d_model)
        g2 = g2.mean(dim=2)  # joint pooling -> (B, T, d_model)
        g_feat = self.temporal_pool_gcn(g2.transpose(1, 2)).squeeze(-1)  # (B, d_model)

        # ----- Branch 2: Spatial smooth + Transformer -----
        # Use adjacency as implicit structure but attention over joints (no A in attn)
        s1 = self.spatial_attn(x)  # (B, T, J, d_model)
        s1 = s1.mean(dim=2)  # joint pooling -> (B, T, d_model)
        t_out = self.temporal_transformer(s1)  # (B, T, d_model)
        t_feat = t_out.mean(dim=1)  # temporal mean -> (B, d_model)

        # ----- Fusion -----
        fused = torch.cat([g_feat, t_feat], dim=-1)  # (B, 2*d_model)
        out = self.fusion_mlp(fused).squeeze(-1)  # (B,)
        return out
