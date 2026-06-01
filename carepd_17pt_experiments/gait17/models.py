from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import H36M17_EDGES


def adjacency(num_joints: int = 17) -> torch.Tensor:
    a = torch.eye(num_joints)
    for i, j in H36M17_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    degree = a.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return a / degree


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return torch.einsum("ij,btjc->btic", a.to(x.device), x)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, dropout: float = 0.2):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.res = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        res = x if self.res is None else self.res(x)
        x = self.gcn(x, a)
        x = self.tcn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return F.relu(x + res)


class STGCNRegressor(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 64):
        super().__init__()
        self.register_buffer("A", adjacency())
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(in_channels, hidden),
                STGCNBlock(hidden, hidden),
                STGCNBlock(hidden, hidden * 2),
            ]
        )
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, self.A)
        x = x.mean(dim=(1, 2))
        return self.head(x).squeeze(-1)


class TemporalCNNRegressor(nn.Module):
    def __init__(self, in_channels: int, num_joints: int = 17, hidden: int = 128):
        super().__init__()
        input_dim = in_channels * num_joints
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, j, c = x.shape
        x = x.reshape(b, t, j * c).transpose(1, 2)
        return self.head(self.net(x)).squeeze(-1)


class OursHybridEncoder(nn.Module):
    def __init__(self, in_channels: int = 9, hidden: int = 128):
        super().__init__()
        self.register_buffer("A", adjacency())
        self.gcn1 = GraphConv(in_channels, 64)
        self.gcn2 = GraphConv(64, hidden)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.joint_attn = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.gcn1(x, self.A)))
        x = F.relu(self.norm2(self.gcn2(x, self.A)))
        weights = torch.softmax(self.joint_attn(x), dim=2)
        x = (x * weights).sum(dim=2)
        return self.encoder(x).mean(dim=1)


class OursMLPOnly(nn.Module):
    """Architecture-ablation baseline: no graph, no joint attention, no Transformer."""

    def __init__(self, in_channels: int = 9, hidden: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(1, 2))
        return 3.0 * torch.sigmoid(self.head(x).squeeze(-1))


class OursGraphConvMLP(nn.Module):
    """Architecture ablation: GraphConv encoder with temporal/joint mean pooling."""

    def __init__(self, in_channels: int = 9, hidden: int = 128):
        super().__init__()
        self.register_buffer("A", adjacency())
        self.gcn1 = GraphConv(in_channels, 64)
        self.gcn2 = GraphConv(64, hidden)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.gcn1(x, self.A)))
        x = F.relu(self.norm2(self.gcn2(x, self.A)))
        x = x.mean(dim=(1, 2))
        return 3.0 * torch.sigmoid(self.head(x).squeeze(-1))


class OursGraphJointAttnMLP(nn.Module):
    """Architecture ablation: GraphConv + joint attention without Temporal Transformer."""

    def __init__(self, in_channels: int = 9, hidden: int = 128):
        super().__init__()
        self.register_buffer("A", adjacency())
        self.gcn1 = GraphConv(in_channels, 64)
        self.gcn2 = GraphConv(64, hidden)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(hidden)
        self.joint_attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.gcn1(x, self.A)))
        x = F.relu(self.norm2(self.gcn2(x, self.A)))
        weights = torch.softmax(self.joint_attn(x), dim=2)
        x = (x * weights).sum(dim=2).mean(dim=1)
        return 3.0 * torch.sigmoid(self.head(x).squeeze(-1))


class OursGait17(nn.Module):
    def __init__(self, in_channels: int = 9, hidden: int = 128):
        super().__init__()
        self.encoder = OursHybridEncoder(in_channels, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return 3.0 * torch.sigmoid(self.head(z).squeeze(-1))


class ConvBNLeaky1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseBNLeaky(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.linear(x)))


class LuOfficialFeatureModule(nn.Module):
    """PyTorch port of the official DD-Net feature module used by Lu et al."""

    def __init__(self, frame_l: int, joint_n: int, joint_d: int, feat_d: int, filters: int):
        super().__init__()
        pose_dim = joint_n * joint_d
        self.frame_l = frame_l
        self.joint_n = joint_n
        self.joint_d = joint_d

        self.m_branch = nn.Sequential(
            ConvBNLeaky1D(feat_d, filters * 2, 1),
            nn.Dropout1d(0.1),
            ConvBNLeaky1D(filters * 2, filters, 3),
            nn.Dropout1d(0.1),
            ConvBNLeaky1D(filters, filters, 1),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
        )
        self.slow_branch = nn.Sequential(
            ConvBNLeaky1D(pose_dim, filters * 2, 1),
            nn.Dropout1d(0.1),
            ConvBNLeaky1D(filters * 2, filters, 3),
            nn.Dropout1d(0.1),
            ConvBNLeaky1D(filters, filters, 1),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
        )
        self.fast_branch = nn.Sequential(
            ConvBNLeaky1D(pose_dim, filters * 2, 1),
            nn.Dropout1d(0.1),
            ConvBNLeaky1D(filters * 2, filters, 3),
            nn.Dropout1d(0.1),
            ConvBNLeaky1D(filters, filters, 1),
            nn.Dropout1d(0.1),
        )
        self.block1 = nn.Sequential(
            ConvBNLeaky1D(filters * 3, filters * 2, 3),
            ConvBNLeaky1D(filters * 2, filters * 2, 3),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
        )
        self.block2 = nn.Sequential(
            ConvBNLeaky1D(filters * 2, filters * 4, 3),
            ConvBNLeaky1D(filters * 4, filters * 4, 3),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.1),
        )
        self.block3 = nn.Sequential(
            ConvBNLeaky1D(filters * 4, filters * 8, 3),
            ConvBNLeaky1D(filters * 8, filters * 8, 3),
            nn.Dropout1d(0.1),
        )

    @staticmethod
    def _resize_pose_nearest(x: torch.Tensor, target_t: int, target_j: int) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(target_t, target_j), mode="nearest")
        return x.permute(0, 2, 3, 1)

    def _pose_motion(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slow = p[:, 1:] - p[:, :-1]
        slow = self._resize_pose_nearest(slow, p.shape[1], p.shape[2])
        slow = slow.flatten(2).transpose(1, 2)

        fast_pose = p[:, ::2]
        fast = fast_pose[:, 1:] - fast_pose[:, :-1]
        fast = self._resize_pose_nearest(fast, fast_pose.shape[1], fast_pose.shape[2])
        fast = fast.flatten(2).transpose(1, 2)
        return slow, fast

    def forward(self, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        slow, fast = self._pose_motion(p)
        m = m.transpose(1, 2)
        x = self.m_branch(m)
        x_slow = self.slow_branch(slow)
        x_fast = self.fast_branch(fast)
        min_t = min(x.shape[-1], x_slow.shape[-1], x_fast.shape[-1])
        x = torch.cat([x[..., :min_t], x_slow[..., :min_t], x_fast[..., :min_t]], dim=1)
        return self.block3(self.block2(self.block1(x)))


class LuOFDDNetOfficial(nn.Module):
    """PyTorch port of the official Lu et al. DD-Net/OF-DDNet Keras model."""

    def __init__(
        self,
        frame_l: int = 390,
        num_joints: int = 17,
        joint_dim: int = 3,
        filters: int = 16,
        num_classes: int = 4,
    ):
        super().__init__()
        feat_dim = num_joints * (num_joints - 1) // 2
        self.fm = LuOfficialFeatureModule(frame_l, num_joints, joint_dim, feat_dim, filters)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            DenseBNLeaky(filters * 8, 128),
            nn.Dropout(0.5),
            DenseBNLeaky(128, 128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _jcd(x: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, x)
        idx = torch.triu_indices(x.shape[2], x.shape[2], offset=1, device=x.device)
        return dist[:, :, idx[0], idx[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self._jcd(x)
        z = self.fm(m, x)
        z = self.pool(z).squeeze(-1)
        return self.head(z)


def expected_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    scores = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (torch.softmax(logits, dim=-1) * scores).sum(dim=-1)


def ordinal_focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25, beta: float = 0.2) -> torch.Tensor:
    target = y.round().clamp(0, logits.shape[-1] - 1).long()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    focal = alpha * (1.0 - probs).pow(gamma)
    pred_class = probs.argmax(dim=-1)
    ordinal_weight = (target - pred_class).abs().float() / max(logits.shape[-1] - 1, 1)
    ce = F.nll_loss(log_probs, target, reduction="none")
    pt = probs.gather(1, target[:, None]).squeeze(1)
    focal_term = alpha * (1.0 - pt).pow(gamma) * ce
    return (focal_term + beta * ordinal_weight * ce).mean()


def make_model(name: str, in_channels: int):
    if name == "temporal_cnn":
        return TemporalCNNRegressor(in_channels=in_channels)
    if name == "ours_mlp":
        return OursMLPOnly(in_channels=in_channels)
    if name == "ours_gcn_mlp":
        return OursGraphConvMLP(in_channels=in_channels)
    if name == "ours_gcn_attn_mlp":
        return OursGraphJointAttnMLP(in_channels=in_channels)
    if name == "ours":
        return OursGait17(in_channels=in_channels)
    if name == "stgcn":
        return STGCNRegressor(in_channels=in_channels)
    if name == "lu_ofddnet_official":
        return LuOFDDNetOfficial()
    raise ValueError(f"Unsupported model: {name}")
