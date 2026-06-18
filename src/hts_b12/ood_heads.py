"""OOD-oriented compositional heads for HtS-B12.

The original HtS-B12 classifier uses one dense class logit per possible target.
For strict length extrapolation this is a poor inductive bias: if training labels
only cover 0..128, classes 129..200 receive little/no direct positive gradient.

The digit head predicts a number compositionally as hundreds/tens/ones.  This is
not a hand-coded counter: the encoder still has to infer the target value from
input tokens and task id.  The head only changes the *label parameterization* so
that unseen integers can be composed from digit factors learned on seen integers.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import HtSB12Config
from .layers import SinusoidalPosition
from .modeling import HtSB12EncoderLayer


class DigitOutputMixin:
    """Mixin utilities for three-digit integer heads."""

    max_digit_value: int

    @staticmethod
    def split_digits(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels.clamp_min(0)
        hundreds = (labels // 100).clamp(0, 9)
        tens = (labels // 10) % 10
        ones = labels % 10
        return hundreds.long(), tens.long(), ones.long()

    @staticmethod
    def compose_digits(hundreds: torch.Tensor, tens: torch.Tensor, ones: torch.Tensor) -> torch.Tensor:
        return (100 * hundreds + 10 * tens + ones).long()

    def digit_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        label_smoothing: float = 0.0,
        digit_weights: Optional[Tuple[float, float, float]] = None,
    ) -> torch.Tensor:
        h, t, o = self.split_digits(labels)
        # Hundreds head is configured to the needed range, usually 0..2 for max<=256.
        max_h = outputs["hundreds"].size(-1) - 1
        h = h.clamp(max=max_h)
        w = digit_weights if digit_weights is not None else (1.0, 1.0, 1.0)
        return (
            w[0] * F.cross_entropy(outputs["hundreds"], h, label_smoothing=label_smoothing)
            + w[1] * F.cross_entropy(outputs["tens"], t, label_smoothing=label_smoothing)
            + w[2] * F.cross_entropy(outputs["ones"], o, label_smoothing=label_smoothing)
        )

    def predict_digits(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = outputs["hundreds"].argmax(dim=-1)
        t = outputs["tens"].argmax(dim=-1)
        o = outputs["ones"].argmax(dim=-1)
        y = self.compose_digits(h, t, o)
        return y.clamp(0, int(getattr(self, "max_digit_value", 999)))

    def digit_accuracy(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> float:
        pred = self.predict_digits(outputs)
        return float((pred == labels).float().mean().detach().cpu())


class HtSB12DigitClassifier(nn.Module, DigitOutputMixin):
    """HtS-B12 encoder with compositional digit output head.

    This variant is intended for OOD numeric targets such as length/count values
    outside the exact class range observed during training.  Instead of producing
    one independent logit for each class, it predicts hundreds/tens/ones and
    composes an integer at inference time.
    """

    config_class = HtSB12Config

    def __init__(self, config: HtSB12Config, max_digit_value: Optional[int] = None):
        super().__init__()
        self.config = config
        self.max_digit_value = int(max_digit_value if max_digit_value is not None else config.num_classes - 1)
        extra = 1 if config.use_cls_token else 0
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.task_input_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.d_model)) if config.use_cls_token else None
        self.pos = SinusoidalPosition(config.max_length + extra, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([HtSB12EncoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.d_model)

        # Task-aware output adapter.  The task is added here again so the same
        # encoder representation can emit length vs different count targets.
        self.task_head_emb = nn.Embedding(config.num_tasks, config.d_model)
        self.head_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        n_hundreds = max(1, self.max_digit_value // 100 + 1)
        self.hundreds = nn.Linear(config.d_model, n_hundreds)
        self.tens = nn.Linear(config.d_model, 10)
        self.ones = nn.Linear(config.d_model, 10)

        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.task_input_emb.weight, std=0.02)
        nn.init.normal_(self.task_head_emb.weight, std=0.02)
        if self.cls is not None:
            nn.init.normal_(self.cls, std=0.02)

    def encode(
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
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        for layer in self.layers:
            x = layer(x, task_ids, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        if self.config.pool == "mean":
            if attention_mask is None:
                pooled = x.mean(dim=1)
            else:
                mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = x[:, 0]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        task_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled = self.encode(input_ids, task_ids, attention_mask)
        z = self.head_mlp(pooled + self.task_head_emb(task_ids))
        return {"hundreds": self.hundreds(z), "tens": self.tens(z), "ones": self.ones(z)}

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
