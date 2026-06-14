from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import HtSConfig, TransformerConfig
from .layers import HtSB12FFN, StaticFFN


class TokenPosEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d_model: int) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, l = input_ids.shape
        if l > self.max_len:
            raise ValueError(f"Sequence length {l} exceeds max_len={self.max_len}")
        pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        return self.norm(self.token(input_ids) + self.pos(pos))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn: nn.Module, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = ffn
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x, task_ids)
        x = self.ln2(x + self.dropout(ffn_out))
        return x

    def hts_losses(self):
        if hasattr(self.ffn, "hts_losses"):
            return self.ffn.hts_losses()
        dev = next(self.parameters()).device
        z = torch.tensor(0.0, device=dev)
        return z, z, z, z

    def diagnostics(self) -> Dict[str, float]:
        if hasattr(self.ffn, "diagnostics"):
            return self.ffn.diagnostics()
        return {}


class HtSTransformerClassifier(nn.Module):
    """HtS-B12 classifier with true-FFN generated soft-weight updates."""

    def __init__(self, config: HtSConfig) -> None:
        super().__init__()
        self.config = config
        self.emb = TokenPosEmbedding(config.vocab_size, config.max_len, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                HtSB12FFN(
                    d_model=config.d_model,
                    dim_ff=config.dim_ff,
                    num_tasks=config.num_tasks,
                    task_dim=config.task_dim,
                    rank_main=config.rank_main,
                    rank_corr=config.rank_corr,
                    alpha_max=config.alpha_max,
                    target_min=config.target_min,
                    target_max=config.target_max,
                    tune_scale=config.tune_scale,
                    gate_bias=config.gate_bias,
                    task_offset_scale=config.task_offset_scale,
                    corr_alpha_max=config.corr_alpha_max,
                    corr_gain=config.corr_gain,
                    ratio_ceiling=config.ratio_ceiling,
                    corr_ceiling=config.corr_ceiling,
                    correction_mode=config.correction_mode,
                    name=f"hts_l{i}",
                ),
                config.dropout,
            ) for i in range(config.n_layers)
        ])
        self.head = nn.Linear(config.d_model, config.output_dim)

    def forward(self, input_ids: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        for block in self.blocks:
            x = block(x, task_ids)
        pooled = x[:, 0]  # CLS-like first token
        return self.head(pooled)

    def hts_losses(self):
        vals = []
        for block in self.blocks:
            vals.append(block.hts_losses())
        if not vals:
            dev = next(self.parameters()).device
            z = torch.tensor(0.0, device=dev)
            return z, z, z, z
        return tuple(sum(v[i] for v in vals) / len(vals) for i in range(4))

    def diagnostics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, block in enumerate(self.blocks):
            for k, v in block.diagnostics().items():
                out[f"block{i}.{k}"] = v
        return out


class StaticTransformerClassifier(nn.Module):
    """Static Transformer baseline with identical interface."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.emb = TokenPosEmbedding(config.vocab_size, config.max_len, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                StaticFFN(config.d_model, config.dim_ff, config.dropout),
                config.dropout,
            ) for _ in range(config.n_layers)
        ])
        self.head = nn.Linear(config.d_model, config.output_dim)

    def forward(self, input_ids: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        for block in self.blocks:
            x = block(x, task_ids)
        return self.head(x[:, 0])

    def hts_losses(self):
        dev = next(self.parameters()).device
        z = torch.tensor(0.0, device=dev)
        return z, z, z, z

    def diagnostics(self) -> Dict[str, float]:
        return {}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
