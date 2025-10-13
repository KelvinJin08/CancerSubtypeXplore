# ml_backend/models_nn.py
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== 1) Baseline: Simple Linear ==========
class SimpleLinear(nn.Module):
    """
    Baseline linear classifier with Xavier init.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        return self.fc(x)


# ========== 2) MLP ==========
def _get_act(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unknown activation: {name}")


class SimpleMLP(nn.Module):
    """
    A simple MLP for tabular features.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        activation: str = "relu",
        batch_norm: bool = True,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        layers: List[nn.Module] = []
        prev = input_dim
        act = _get_act(activation)

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, num_classes)

        # Xavier init for stability
        def _xavier(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        z = self.backbone(x)
        logits = self.head(z)
        return logits


# ========== 3) Attention (lightweight) ==========
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.drop(ff_out))
        return x


class AttentionNet(nn.Module):
    """
    Treat each feature as a token, with a shared projection to d_model + learned position embedding.
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        max_len: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        L = max_len or input_dim

        # project scalar feature -> d_model (shared)
        self.scalar_proj = nn.Linear(1, d_model)
        # learned positional embedding for each feature index
        self.pos_embed = nn.Parameter(torch.zeros(1, L, d_model))

        blocks = [TransformerBlock(d_model, n_heads, ff_multiplier, dropout) for _ in range(n_layers)]
        self.encoder = nn.Sequential(*blocks)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        B, F = x.shape
        # [B, F, 1] -> [B, F, d_model]
        z = self.scalar_proj(x.unsqueeze(-1))
        # add pos embed (crop in case F < L)
        z = z + self.pos_embed[:, :F, :]
        # encode
        z = self.encoder(z)
        # pool (mean)
        z = self.norm(z.mean(dim=1))
        logits = self.head(z)
        return logits


# ========== 4) Factory ==========
def build_model(cfg: dict) -> nn.Module:
    kind = cfg["kind"]
    if kind == "linear":
        return SimpleLinear(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
        )
    elif kind == "simple_mlp":
        return SimpleMLP(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            hidden_dims=cfg.get("hidden_dims", [256, 128, 64]),
            dropout=cfg.get("dropout", 0.2),
            activation=cfg.get("activation", "relu"),
            batch_norm=cfg.get("batch_norm", True),
        )
    elif kind == "attention":
        return AttentionNet(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 2),
            ff_multiplier=cfg.get("ff_multiplier", 4),
            dropout=cfg.get("dropout", 0.1),
            max_len=cfg.get("max_len"),
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

