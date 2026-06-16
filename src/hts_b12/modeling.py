"""PyTorch models exposed by the hts_b12 package."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .config import HtSB12Config
from .layers import HtSB12FFN, SinusoidalPosition


def _masked_sequence_mean(x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if attention_mask is None:
        return x.mean(dim=1)
    mask = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class HtSB12EncoderLayer(nn.Module):
    def __init__(self, config: HtSB12Config, layer_id: int):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.ffn = HtSB12FFN(
            d_model=config.d_model,
            dim_ff=config.dim_ff,
            num_tasks=config.num_tasks,
            task_dim=config.task_dim,
            rank_main=config.rank_main,
            rank_corr=config.rank_corr,
            dropout=config.dropout,
            alpha_max=config.alpha_max,
            target_min=config.target_min,
            target_max=config.target_max,
            corr_alpha_max=config.corr_alpha_max,
            corr_gain=config.corr_gain,
            task_offset_scale=config.task_offset_scale,
            ratio_ceiling=config.ratio_ceiling,
            corr_ceiling=config.corr_ceiling,
            name=f"layer{layer_id}_b12",
        )

    def forward(self, x: torch.Tensor, task: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        valid_mask = None if key_padding_mask is None else (~key_padding_mask).to(dtype=x.dtype)
        if self.config.norm_first:
            y = self.norm1(x)
            attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + self.dropout(attn_out)
            ffn_out = self.ffn(self.norm2(x), task, valid_mask=valid_mask)
            x = x + self.dropout(ffn_out)
            return x

        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x, task, valid_mask=valid_mask)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class HtSB12Classifier(nn.Module):
    """HtS-B12 sequence classifier.

    Parameters
    ----------
    config:
        Model hyperparameters. The most important HtS controls are
        ``rank_main``, ``rank_corr``, ``task_dim``, ``target_min/max`` and
        ``task_offset_scale``.

    Forward input
    -------------
    ``input_ids``: LongTensor of shape ``[batch, seq_len]``.
    ``task_ids``: LongTensor of shape ``[batch]``.
    ``attention_mask``: optional mask with 1 for valid tokens and 0 for padding.
    """

    config_class = HtSB12Config

    def __init__(self, config: HtSB12Config):
        super().__init__()
        self.config = config
        extra = 1 if config.use_cls_token else 0
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Direct task signal at input level.  The FFN routers still receive task ids,
        # but this embedding lets attention/CLS pooling separate task families early.
        self.task_input_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model)) if config.use_cls_token else None
        self.pos = SinusoidalPosition(config.max_length + extra, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([HtSB12EncoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.task_input_emb.weight, std=0.02)
        if self.cls is not None:
            nn.init.normal_(self.cls, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_emb(input_ids) + self.task_input_emb(task_ids)[:, None, :]
        if self.cls is not None:
            cls = self.cls.expand(input_ids.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1)
        x = self.dropout(self.pos(x))
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        for layer in self.layers:
            x = layer(x, task_ids, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        if self.config.pool == "mean":
            pooled = _masked_sequence_mean(x, attention_mask)
        else:
            pooled = x[:, 0]
        return self.head(pooled)

    def hts_regularizers(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        regs = [layer.ffn.hts_regularizers() for layer in self.layers]
        cols = list(zip(*regs))
        return tuple(torch.stack(list(c)).mean() for c in cols)  # type: ignore[return-value]

    def hts_diagnostics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, layer in enumerate(self.layers):
            for k, v in layer.ffn.diagnostics().items():
                out[f"l{i}_{k}"] = v
        return out

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "HtSB12Classifier":
        path = Path(path)
        config = HtSB12Config.from_pretrained(path)
        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=map_location))
        return model


class TransformerClassifier(nn.Module):
    """Small Transformer baseline with an API matching HtSB12Classifier."""

    def __init__(self, config: HtSB12Config):
        super().__init__()
        self.config = config
        extra = 1 if config.use_cls_token else 0
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.task_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model)) if config.use_cls_token else None
        self.pos = SinusoidalPosition(config.max_length + extra, config.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)

    def forward(self, input_ids: torch.Tensor, task_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_emb(input_ids) + self.task_emb(task_ids)[:, None, :]
        if self.cls is not None:
            x = torch.cat([self.cls.expand(input_ids.size(0), -1, -1), x], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1)
        x = self.pos(x)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        pooled = x[:, 0] if self.config.pool != "mean" else _masked_sequence_mean(x, attention_mask)
        return self.head(pooled)
