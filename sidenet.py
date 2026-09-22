"""SideNet point-set models for per-cone boundary-side classification."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    """Sin/cos positional encoding for x, y coordinates."""

    def __init__(self, d_model=32, max_freq=8):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq

    def forward(self, x, y):
        orig_shape = x.shape
        x_flat = x.reshape(-1, 1)
        y_flat = y.reshape(-1, 1)

        encodings = []
        for i in range(self.max_freq):
            freq = math.pi * (2**i)
            encodings.append(torch.sin(freq * x_flat))
            encodings.append(torch.cos(freq * x_flat))
            encodings.append(torch.sin(freq * y_flat))
            encodings.append(torch.cos(freq * y_flat))

        enc = torch.cat(encodings, dim=-1)
        if enc.shape[-1] > self.d_model:
            enc = enc[:, : self.d_model]
        elif enc.shape[-1] < self.d_model:
            enc = F.pad(enc, (0, self.d_model - enc.shape[-1]))

        if len(orig_shape) > 1:
            return enc.reshape(*orig_shape, self.d_model)
        return enc.reshape(-1, self.d_model)


# ═══════════════════════════════════════════════════════════════════════════
# Transformer SideNet (original)
# ═══════════════════════════════════════════════════════════════════════════


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, key_padding_mask=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SideNet(nn.Module):
    """
    Transformer-based Left/Right classifier.

    Args:
        input_mode: 'xy' (2 dims), 'xyz' (3 dims), 'xyzs' (4 dims with score)
        d_model: token dimension
        nhead: attention heads
        num_layers: encoder blocks
        dim_feedforward: FFN hidden dim
        dropout: attention dropout
    """

    INPUT_DIMS = {"xy": 2, "xyz": 3, "xyzs": 4}

    def __init__(
        self,
        input_mode="xyz",
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()
        if input_mode not in self.INPUT_DIMS:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        if d_model < 8:
            raise ValueError("d_model must be at least 8")
        self.input_mode = input_mode
        self.d_model = d_model
        input_dim = self.INPUT_DIMS[input_mode]

        pos_dim = d_model // 2
        feature_dim = d_model - pos_dim
        self.pos_enc = PositionalEncoding(d_model=pos_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
        )
        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, points, return_logits=True):
        batched = points.dim() == 3
        if not batched:
            points = points.unsqueeze(0)
        B, N, D = points.shape

        x, y = points[..., 0], points[..., 1]
        pos = self.pos_enc(x, y)
        feat = self.input_proj(points)
        tokens = torch.cat([feat, pos], dim=-1)
        tokens = self.input_norm(tokens)

        key_padding_mask = None
        if N > 0:
            is_pad = points.abs().sum(dim=-1) == 0
            if is_pad.any():
                key_padding_mask = is_pad

        for layer in self.layers:
            tokens = layer(tokens, key_padding_mask=key_padding_mask)

        logits = self.head(tokens)

        if not batched:
            logits = logits.squeeze(0)
        if not return_logits:
            return F.softmax(logits, dim=-1)
        return logits

    def predict(self, points):
        return self.forward(points).argmax(dim=-1)

    def predict_probs(self, points):
        return self.forward(points, return_logits=False)


# ═══════════════════════════════════════════════════════════════════════════
# DGCNN (EdgeConv) — better for local geometric classification
# ═══════════════════════════════════════════════════════════════════════════


def knn(x, k):
    """k-nearest neighbors. x: (B, N, D) → idx: (B, N, k)."""
    if x.shape[1] < 2:
        return torch.empty(x.shape[0], x.shape[1], 0, dtype=torch.long, device=x.device)
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = (x**2).sum(dim=-1, keepdim=True)
    pairwise = xx + inner + xx.transpose(2, 1)
    pairwise.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    k = min(k, x.shape[1] - 1)
    return pairwise.topk(k=k, dim=-1, largest=False)[1]


def get_graph_feature(x, k=16):
    """
    EdgeConv feature: for each point, compute (x_j - x_i, x_i) for k neighbors.
    x: (B, N, D) → out: (B, N, k, 2*D)
    """
    B, N, D = x.shape
    if k >= N:
        k = N - 1
    idx = knn(x, k=k)
    B_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, k)

    # Gather neighbor features: (B, N, k, D)
    neighbors = x[B_idx, idx, :]

    # Central point: (B, N, 1, D) → (B, N, k, D)
    central = x.unsqueeze(2).expand(-1, -1, k, -1)

    # Edge feature: (neighbor - central, central)
    edge = torch.cat([neighbors - central, central], dim=-1)  # (B, N, k, 2*D)
    return edge


class EdgeConvBlock(nn.Module):
    """Single EdgeConv block: graph feature → MLP → max pool."""

    def __init__(self, in_dim, out_dim, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        B, N, D = x.shape
        k = min(self.k, N - 1)
        if k < 1:
            isolated = torch.cat([torch.zeros_like(x), x], dim=-1).unsqueeze(2)
            return self.mlp(isolated).squeeze(2)
        edge = get_graph_feature(x, k=k)  # (B, N, k, 2*D)
        edge = self.mlp(edge)  # (B, N, k, out_dim)
        out = edge.max(dim=2)[0]  # (B, N, out_dim)
        return out


class DGCNN(nn.Module):
    """
    DGCNN-style Left/Right classifier using EdgeConv.

    Architecture: input → EdgeConv(64) → EdgeConv(64) → EdgeConv(128) → MLP → logits

    EdgeConv explicitly computes features between each cone and its k nearest
    neighbors. This is a useful local-geometry bias, but it does not by itself
    guarantee track-topology awareness or Left/Right identifiability.

    Args:
        input_dim: number of input features (2, 3, or 4)
        k: number of nearest neighbors for EdgeConv
        hidden_dims: list of EdgeConv output dimensions
        dropout: dropout rate for final MLP
    """

    def __init__(
        self, input_dim=4, k=16, hidden_dims=(64, 64, 128), dropout=0.1, num_classes=2
    ):
        super().__init__()
        self.k = k

        # EdgeConv layers
        dims = [input_dim] + list(hidden_dims)
        self.edge_convs = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.edge_convs.append(EdgeConvBlock(dims[i], dims[i + 1], k=k))

        # Global + local feature → classification
        total_dim = sum(hidden_dims)  # concatenated skip connections
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, points, return_logits=True):
        """
        Args:
            points: (N, D) or (B, N, D)
            return_logits: if True return logits, else probabilities
        Returns:
            (N, 2) or (B, N, 2)
        """
        batched = points.dim() == 3
        if not batched:
            points = points.unsqueeze(0)
        B, N, D = points.shape

        # EdgeConv with skip connections
        features = []
        x = points
        for ec in self.edge_convs:
            x = ec(x)
            features.append(x)

        # Concatenate multi-scale features
        x = torch.cat(features, dim=-1)  # (B, N, total_dim)

        logits = self.classifier(x)

        if not batched:
            logits = logits.squeeze(0)
        if not return_logits:
            return F.softmax(logits, dim=-1)
        return logits

    def predict(self, points):
        return self.forward(points).argmax(dim=-1)

    def predict_probs(self, points):
        return self.forward(points, return_logits=False)


# ═══════════════════════════════════════════════════════════════════════════
# PointNet++ — hierarchical point set abstraction + feature propagation
# ═══════════════════════════════════════════════════════════════════════════


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += (src**2).sum(dim=-1, keepdim=True)
    dist += (dst**2).sum(dim=-1, keepdim=True).permute(0, 2, 1)
    return dist.clamp(min=1e-10)


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    # A deterministic first center keeps validation and inference repeatable.
    centroid = xyz.mean(dim=1, keepdim=True)
    farthest = ((xyz - centroid) ** 2).sum(dim=-1).max(dim=-1)[1]
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].view(B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=-1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    dist = square_distance(new_xyz[:, :, :3], xyz[:, :, :3])
    masked_dist = dist.masked_fill(dist > radius**2, float("inf"))
    k = min(nsample, N)
    neighbor_dist, group_idx = masked_dist.topk(k, dim=-1, largest=False)
    nearest = dist.argmin(dim=-1, keepdim=True).expand(-1, -1, k)
    return torch.where(torch.isfinite(neighbor_dist), group_idx, nearest)


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = nn.Sequential()
        for i, out_ch in enumerate(mlp):
            self.mlp.add_module(f"c{i}", nn.Conv2d(in_channel, out_ch, 1))
            self.mlp.add_module(f"bn{i}", nn.BatchNorm2d(out_ch))
            self.mlp.add_module(f"r{i}", nn.ReLU())
            in_channel = out_ch

    def forward(self, xyz, features):
        B, N, _ = xyz.shape
        npoint = min(self.npoint, N)
        fps_idx = farthest_point_sample(xyz, npoint)
        batch_idx = torch.arange(B, device=xyz.device).unsqueeze(-1)
        new_xyz = xyz[batch_idx, fps_idx, :]

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)

        # Gather grouped xyz: idx (B, npoint, nsample) → xyz (B, N, 3)
        n_neighbors = idx.shape[-1]
        B_idx = (
            torch.arange(B, device=xyz.device)
            .view(B, 1, 1)
            .expand(-1, npoint, n_neighbors)
        )
        grouped_xyz = xyz[B_idx, idx, :] - new_xyz.unsqueeze(
            2
        )  # (B, npoint, nsample, 3)

        if features is not None:
            _, C, _ = features.shape
            # Gather features: idx (B, npoint, nsample) → features (B, C, N)
            features_exp = features.unsqueeze(3).expand(-1, -1, -1, n_neighbors)
            n_idx_f = idx.unsqueeze(1).expand(-1, C, -1, -1)
            grouped_feat = torch.gather(features_exp, 2, n_idx_f)
            grouped_feat = torch.cat(
                [grouped_xyz.permute(0, 3, 1, 2), grouped_feat], dim=1
            )
        else:
            grouped_feat = grouped_xyz.permute(0, 3, 1, 2)

        new_features = self.mlp(grouped_feat).max(dim=-1)[0]
        return new_xyz, new_features


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp = nn.Sequential()
        for i, out_ch in enumerate(mlp):
            self.mlp.add_module(f"c{i}", nn.Conv1d(in_channel, out_ch, 1))
            self.mlp.add_module(f"bn{i}", nn.BatchNorm1d(out_ch))
            self.mlp.add_module(f"r{i}", nn.ReLU())
            in_channel = out_ch

    def forward(self, xyz1, xyz2, feat1, feat2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = feat2.expand(-1, -1, N)
        else:
            dist = square_distance(xyz1[:, :, :3], xyz2[:, :, :3])
            dist, idx = dist.topk(min(3, S), dim=-1, largest=False)
            dist = dist.clamp(min=1e-10)
            weight = 1.0 / dist
            weight = weight / weight.sum(dim=-1, keepdim=True)

            idx_f = idx.unsqueeze(1).expand(-1, feat2.shape[1], -1, -1)
            interp_feat = torch.gather(
                feat2.unsqueeze(-1).expand(-1, -1, -1, min(3, S)), dim=2, index=idx_f
            )
            interpolated = (interp_feat * weight.unsqueeze(1)).sum(dim=-1)

        new_feat = (
            torch.cat([interpolated, feat1], dim=1)
            if feat1 is not None
            else interpolated
        )
        return self.mlp(new_feat)


class PointNet2(nn.Module):
    """
    PointNet++ for Left/Right cone classification.

    SA1: 16 centers, r=5m, 16 neighbors → 64 dims
    SA2:  8 centers, r=10m, 8 neighbors → 128 dims
    FP2: 128+64 → 128 → 64  (back to SA1 level)
    FP1: 64+in_dim → 64  (back to original points)
    Classifier: 64 → 32 → 2
    """

    def __init__(self, input_dim=4, dropout=0.1, num_classes=2):
        super().__init__()
        self.sa1 = SetAbstraction(16, 5.0, 16, input_dim + 3, [32, 32, 64])
        self.sa2 = SetAbstraction(8, 10.0, 8, 64 + 3, [64, 64, 128])
        self.fp2 = FeaturePropagation(128 + 64, [128, 64])
        self.fp1 = FeaturePropagation(64 + input_dim, [64, 64])

        self.classifier = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, num_classes, 1),
        )

    def forward(self, points, return_logits=True):
        batched = points.dim() == 3
        if not batched:
            points = points.unsqueeze(0)
        B, N, D = points.shape

        if D >= 3:
            xyz = points[..., :3]
        else:
            xyz = F.pad(points[..., :2], (0, 1))
        features = points.permute(0, 2, 1)

        xyz1, feat1 = self.sa1(xyz, features)
        xyz2, feat2 = self.sa2(xyz1, feat1)

        feat_sa1 = self.fp2(xyz1, xyz2, feat1, feat2)
        feat_orig = self.fp1(xyz, xyz1, features, feat_sa1)

        logits = self.classifier(feat_orig).permute(0, 2, 1)

        if not batched:
            logits = logits.squeeze(0)
        if not return_logits:
            return F.softmax(logits, dim=-1)
        return logits

    def predict(self, points):
        return self.forward(points).argmax(dim=-1)

    def predict_probs(self, points):
        return self.forward(points, return_logits=False)


# ═══════════════════════════════════════════════════════════════════════════
# PointNet — per-point MLP + global max pool, no hierarchy, no neighbors
# ═══════════════════════════════════════════════════════════════════════════


class PointNet(nn.Module):
    """
    Original PointNet for per-point classification.

    Per-point MLP → global max pool → concat global+local → classifier.

    Unlike PointNet++, this never samples points. Unlike DGCNN, it has no
    explicit neighbor queries — it relies on the global max pool for context.
    """

    def __init__(self, input_dim=4, dropout=0.1, num_classes=2):
        super().__init__()

        self.local_mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(1024 + 128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, points, return_logits=True):
        batched = points.dim() == 3
        if not batched:
            points = points.unsqueeze(0)
        B, N, D = points.shape
        x = points.permute(0, 2, 1)

        local = self.local_mlp(x)
        global_feat = self.global_mlp(local).max(dim=-1, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, N)

        combined = torch.cat([local, global_feat], dim=1)
        logits = self.classifier(combined).permute(0, 2, 1)

        if not batched:
            logits = logits.squeeze(0)
        if not return_logits:
            return F.softmax(logits, dim=-1)
        return logits

    def predict(self, points):
        return self.forward(points).argmax(dim=-1)

    def predict_probs(self, points):
        return self.forward(points, return_logits=False)


def build_model(model_cfg, data_cfg):
    """Build a SideNet model from the YAML model/data sections."""
    architecture = model_cfg.get("architecture", "dgcnn")
    input_mode = data_cfg.get("input_mode", "xyz")
    if input_mode not in SideNet.INPUT_DIMS:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    input_dim = SideNet.INPUT_DIMS[input_mode]
    num_classes = int(data_cfg.get("num_classes", 2))

    if architecture == "dgcnn":
        return DGCNN(
            input_dim=input_dim,
            k=model_cfg.get("dgcnn_k", 16),
            hidden_dims=model_cfg.get("dgcnn_dims", [64, 64, 128]),
            dropout=model_cfg.get("dropout", 0.1),
            num_classes=num_classes,
        )
    if architecture == "pointnet2":
        return PointNet2(
            input_dim=input_dim,
            dropout=model_cfg.get("dropout", 0.1),
            num_classes=num_classes,
        )
    if architecture == "pointnet":
        return PointNet(
            input_dim=input_dim,
            dropout=model_cfg.get("dropout", 0.1),
            num_classes=num_classes,
        )
    if architecture == "transformer":
        return SideNet(
            input_mode=input_mode,
            d_model=model_cfg.get("d_model", 64),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 128),
            dropout=model_cfg.get("dropout", 0.1),
            num_classes=num_classes,
        )
    raise ValueError(f"Architecture {architecture!r} does not define a trainable model")
