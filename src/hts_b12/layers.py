"""Core layers for HtS-B12."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPosition(nn.Module):
    def __init__(self, max_length: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


class TaskConditionedLowRank(nn.Module):
    """Input/task-conditioned low-rank generated update.

    Given token states ``x`` and task ids, this module returns a generated delta
    with the same last dimension as a target linear output. It implements the
    low-rank HtS idea without materializing a full dense matrix per example.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        task_dim: int,
        num_tasks: int,
        hidden: Optional[int] = None,
        tune_scale: float = 0.25,
        name: str = "delta",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.name = name
        hidden = hidden or max(32, task_dim * 2)

        self.task_emb = nn.Embedding(num_tasks, task_dim)
        self.a = nn.Linear(in_features, rank, bias=False)
        self.b = nn.Linear(rank, out_features, bias=False)
        self.router = nn.Sequential(
            nn.Linear(in_features + task_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, rank),
        )
        self.tune_scale = tune_scale
        nn.init.normal_(self.a.weight, std=0.02)
        nn.init.normal_(self.b.weight, std=0.02)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self._last: Dict[str, float] = {}

    def forward(self, x: torch.Tensor, task: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ctx is None:
            ctx = x.mean(dim=1)
        te = self.task_emb(task)
        coeff = 1.0 + self.tune_scale * torch.tanh(self.router(torch.cat([ctx, te], dim=-1)))
        z = self.a(x) * coeff[:, None, :]
        out = self.b(z)
        self._last = {
            f"{self.name}_rank": float(self.rank),
            f"{self.name}_coeff_abs": float(coeff.detach().abs().mean()),
            f"{self.name}_coeff_std": float(coeff.detach().std()),
        }
        return out

    def budget_tensor(self) -> torch.Tensor:
        # Smooth proxy. Kept as Tensor for regularization compatibility.
        return self.a.weight.abs().mean() + self.b.weight.abs().mean()

    def diagnostics(self) -> Dict[str, float]:
        return dict(self._last)




def _masked_mean(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over tokens, ignoring padding when a valid-token mask is supplied."""
    if valid_mask is None:
        return x.mean(dim=1)
    m = valid_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _masked_norm_mean(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean token norm with optional padding mask, returned as [B,1,1]."""
    n = x.norm(dim=-1, keepdim=True)
    if valid_mask is None:
        return n.mean(dim=1, keepdim=True)
    m = valid_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
    return (n * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)

class HtSB12FFN(nn.Module):
    """B12 true-FFN generated-computation block.

    The update is applied inside the real FFN path:

    ``h = GELU(W1(x) + main_ratio_delta + input_correction_delta)``
    ``y = W2(h) + main_ratio_delta``

    B12 adds task-specific router offsets and an optional margin-oriented loss
    at model level. The soft deltas are generated dynamically by hard routers.
    """

    def __init__(
        self,
        d_model: int,
        dim_ff: int,
        num_tasks: int,
        task_dim: int,
        rank_main: int,
        rank_corr: int,
        dropout: float = 0.1,
        alpha_max: float = 1.20,
        target_min: float = 0.25,
        target_max: float = 0.90,
        corr_alpha_max: float = 0.55,
        corr_gain: float = 6.0,
        task_offset_scale: float = 0.30,
        ratio_ceiling: float = 0.95,
        corr_ceiling: float = 0.35,
        name: str = "b12ffn",
    ) -> None:
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.dim_ff = dim_ff
        self.alpha_max = alpha_max
        self.target_min = target_min
        self.target_max = target_max
        self.corr_alpha_max = corr_alpha_max
        self.corr_gain = corr_gain
        self.task_offset_scale = task_offset_scale
        self.ratio_ceiling = ratio_ceiling
        self.corr_ceiling = corr_ceiling

        self.base_l1 = nn.Linear(d_model, dim_ff)
        self.base_l2 = nn.Linear(dim_ff, d_model)
        self.task_emb = nn.Embedding(num_tasks, task_dim)
        self.router = nn.Sequential(
            nn.Linear(d_model + task_dim, max(64, task_dim * 2)),
            nn.GELU(),
            nn.Linear(max(64, task_dim * 2), 6),
        )
        self.task_router_offset = nn.Embedding(num_tasks, 6)
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        nn.init.zeros_(self.task_router_offset.weight)

        self.main1 = TaskConditionedLowRank(d_model, dim_ff, rank_main, task_dim, num_tasks, name=f"{name}_main1")
        self.main2 = TaskConditionedLowRank(dim_ff, d_model, rank_main, task_dim, num_tasks, name=f"{name}_main2")
        self.corr1 = TaskConditionedLowRank(d_model, dim_ff, rank_corr, task_dim, num_tasks, tune_scale=0.20, name=f"{name}_corr1")
        self.dropout = nn.Dropout(dropout)
        self._last: Dict[str, float] = {}
        self._budget = torch.tensor(0.0)
        self._binary = torch.tensor(0.0)
        self._ratio_penalty = torch.tensor(0.0)
        self._task_offset_l2 = torch.tensor(0.0)

    @staticmethod
    def _targeted(
        base: torch.Tensor,
        raw: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        eps = 1e-6
        base_norm = _masked_norm_mean(base.detach(), valid_mask)
        raw_norm = _masked_norm_mean(raw, valid_mask) + eps
        scale = target[:, None, :] * base_norm / raw_norm
        # Guard against rare enormous scales from unlucky near-zero raw deltas.
        scale = scale.clamp(max=8.0)
        return raw * scale

    def forward(self, x: torch.Tensor, task: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = _masked_mean(x, valid_mask)
        te = self.task_emb(task)
        rr = self.router(torch.cat([ctx, te], dim=-1))
        off = self.task_offset_scale * torch.tanh(self.task_router_offset(task))
        rr = rr + off

        gate = torch.sigmoid(rr[:, 0:1])
        alpha = self.alpha_max * torch.sigmoid(rr[:, 1:2])
        target1 = self.target_min + (self.target_max - self.target_min) * torch.sigmoid(rr[:, 2:3])
        target2 = self.target_min + (self.target_max - self.target_min) * torch.sigmoid(rr[:, 3:4])
        cgate = torch.sigmoid(rr[:, 4:5])
        calpha = self.corr_alpha_max * torch.sigmoid(rr[:, 5:6])

        base1 = self.base_l1(x)
        raw_main1 = self.main1(x, task, ctx)
        main1 = gate[:, None, :] * alpha[:, None, :] * self._targeted(base1, raw_main1, target1, valid_mask)
        corr1 = cgate[:, None, :] * calpha[:, None, :] * self.corr_gain * self.corr1(x, task, ctx)
        h = F.gelu(base1 + main1 + corr1)
        h = self.dropout(h)

        ctx2 = _masked_mean(h, valid_mask)
        base2 = self.base_l2(h)
        raw_main2 = self.main2(h, task, ctx2)
        main2 = gate[:, None, :] * alpha[:, None, :] * self._targeted(base2, raw_main2, target2, valid_mask)
        y = base2 + main2

        eps = 1e-6
        ratio1 = (_masked_norm_mean(main1 + corr1, valid_mask).mean() / (_masked_norm_mean(base1.detach(), valid_mask).mean() + eps))
        ratio2 = (_masked_norm_mean(main2, valid_mask).mean() / (_masked_norm_mean(base2.detach(), valid_mask).mean() + eps))
        corr_ratio = (_masked_norm_mean(corr1, valid_mask).mean() / (_masked_norm_mean(base1.detach(), valid_mask).mean() + eps))
        ratio = 0.5 * (ratio1 + ratio2)

        budget = gate.mean() * 0.5 * (self.main1.budget_tensor() + self.main2.budget_tensor())
        budget = budget + 0.25 * cgate.mean() * self.corr1.budget_tensor()
        binary = (gate * (1.0 - gate)).mean() + 0.5 * (cgate * (1.0 - cgate)).mean()
        ratio_penalty = F.relu(ratio - self.ratio_ceiling).pow(2) + 0.5 * F.relu(corr_ratio - self.corr_ceiling).pow(2)
        task_offset_l2 = off.pow(2).mean()

        self._budget = budget
        self._binary = binary
        self._ratio_penalty = ratio_penalty
        self._task_offset_l2 = task_offset_l2
        self._last = {
            f"{self.name}_gate_main": float(gate.detach().mean()),
            f"{self.name}_alpha_main": float(alpha.detach().mean()),
            f"{self.name}_gate_corr": float(cgate.detach().mean()),
            f"{self.name}_alpha_corr": float(calpha.detach().mean()),
            f"{self.name}_delta_base_ratio": float(ratio.detach()),
            f"{self.name}_corr_ratio": float(corr_ratio.detach()),
            f"{self.name}_target1": float(target1.detach().mean()),
            f"{self.name}_target2": float(target2.detach().mean()),
            f"{self.name}_task_offset_abs": float(off.detach().abs().mean()),
            f"{self.name}_budget": float(budget.detach()),
            f"{self.name}_binary": float(binary.detach()),
            f"{self.name}_ratio_penalty": float(ratio_penalty.detach()),
        }
        self._last.update(self.main1.diagnostics())
        self._last.update(self.main2.diagnostics())
        self._last.update(self.corr1.diagnostics())
        return y

    def hts_regularizers(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._budget, self._binary, self._ratio_penalty, self._task_offset_l2

    def diagnostics(self) -> Dict[str, float]:
        return dict(self._last)
