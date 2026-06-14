from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratedDiagonalLinear(nn.Module):
    """Generated low-rank linear map: (x A^T) diag(coeff) B^T.

    The generator is hard-weighted; the per-sample coefficients are soft weights
    generated from task id and input context. No generated parameter is stored as
    a trainable nn.Parameter directly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        num_tasks: int,
        task_dim: int,
        ctx_dim: int,
        tune_scale: float = 0.34,
        name: str = "gen",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.name = name
        self.A = nn.Parameter(torch.randn(rank, in_features) * (in_features ** -0.5))
        self.B = nn.Parameter(torch.randn(out_features, rank) * (rank ** -0.5))
        self.task_s = nn.Embedding(num_tasks, rank)
        self.task_m = nn.Embedding(num_tasks, rank)
        self.tuner = nn.Sequential(
            nn.Linear(ctx_dim + task_dim, max(16, task_dim * 2)),
            nn.GELU(),
            nn.Linear(max(16, task_dim * 2), rank),
        )
        self.tune_scale = tune_scale
        nn.init.normal_(self.task_s.weight, 0.0, 0.02)
        nn.init.constant_(self.task_m.weight, 0.2)
        self.last: Dict[str, float] = {}

    def coefficients(self, task_ids: torch.Tensor, ctx: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        s_task = self.task_s(task_ids)
        m = torch.sigmoid(self.task_m(task_ids))
        s_tune = self.tune_scale * torch.tanh(self.tuner(torch.cat([ctx, task_emb], dim=-1)))
        coeff = m * (s_task + s_tune)
        with torch.no_grad():
            self.last = {
                f"{self.name}_coeff_abs": float(coeff.detach().abs().mean()),
                f"{self.name}_rank_eff": float((m.detach() > 0.5).float().sum(dim=-1).mean()),
                f"{self.name}_mask_mean": float(m.detach().mean()),
            }
        return coeff

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor, ctx: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        coeff = self.coefficients(task_ids, ctx, task_emb)  # [B, R]
        z = torch.matmul(x, self.A.t())                    # [B, L, R]
        z = z * coeff[:, None, :]
        return torch.matmul(z, self.B.t())                 # [B, L, O]

    def budget_tensor(self, task_ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.task_m(task_ids)).mean()

    def diagnostics(self) -> Dict[str, float]:
        return dict(self.last)


class RatioRouter(nn.Module):
    """Router that generates gate, alpha and target delta/base ratios."""

    def __init__(
        self,
        ctx_dim: int,
        task_dim: int,
        num_tasks: int,
        alpha_max: float,
        target_min: float,
        target_max: float,
        gate_bias: float = -0.06,
        task_offset_scale: float = 0.30,
        corr_alpha_max: float = 0.55,
    ) -> None:
        super().__init__()
        self.alpha_max = alpha_max
        self.corr_alpha_max = corr_alpha_max
        self.target_min = target_min
        self.target_max = target_max
        self.task_offset_scale = task_offset_scale
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + task_dim, max(32, ctx_dim)),
            nn.GELU(),
            nn.Linear(max(32, ctx_dim), 6),
        )
        self.task_offset = nn.Embedding(num_tasks, 6)
        nn.init.zeros_(self.task_offset.weight)
        # Encourage non-zero but controlled gates at initialization.
        with torch.no_grad():
            self.net[-1].bias.zero_()
            self.net[-1].bias[0] = gate_bias
            self.net[-1].bias[4] = gate_bias - 0.2

    def forward(self, ctx: torch.Tensor, task_emb: torch.Tensor, task_ids: torch.Tensor):
        raw = self.net(torch.cat([ctx, task_emb], dim=-1))
        raw = raw + self.task_offset_scale * torch.tanh(self.task_offset(task_ids))
        gate = torch.sigmoid(raw[:, 0:1])
        alpha = self.alpha_max * torch.sigmoid(raw[:, 1:2])
        target1 = self.target_min + (self.target_max - self.target_min) * torch.sigmoid(raw[:, 2:3])
        target2 = self.target_min + (self.target_max - self.target_min) * torch.sigmoid(raw[:, 3:4])
        cgate = torch.sigmoid(raw[:, 4:5])
        calpha = self.corr_alpha_max * torch.sigmoid(raw[:, 5:6])
        return gate, alpha, target1, target2, cgate, calpha, raw


class HtSB12FFN(nn.Module):
    """B12-style true FFN soft-weight update.

    Main path: ratio-controlled soft update in both FFN linear maps.
    Correction path: small free update at the first FFN map, which empirically
    converts lower loss into better accuracy in short CPU-scale benchmarks.
    """

    def __init__(
        self,
        d_model: int,
        dim_ff: int,
        num_tasks: int,
        task_dim: int,
        rank_main: int = 5,
        rank_corr: int = 2,
        alpha_max: float = 1.18,
        target_min: float = 0.34,
        target_max: float = 0.90,
        tune_scale: float = 0.34,
        gate_bias: float = -0.06,
        task_offset_scale: float = 0.30,
        corr_alpha_max: float = 0.55,
        corr_gain: float = 0.55,
        ratio_ceiling: float = 1.35,
        corr_ceiling: float = 0.55,
        correction_mode: str = "input",
        name: str = "hts_ffn",
    ) -> None:
        super().__init__()
        self.name = name
        self.dim_ff = dim_ff
        self.corr_gain = corr_gain
        self.ratio_ceiling = ratio_ceiling
        self.corr_ceiling = corr_ceiling
        self.correction_mode = correction_mode
        self.task_emb = nn.Embedding(num_tasks, task_dim)
        self.base_l1 = nn.Linear(d_model, dim_ff)
        self.base_l2 = nn.Linear(dim_ff, d_model)
        self.router = RatioRouter(d_model, task_dim, num_tasks, alpha_max, target_min, target_max, gate_bias, task_offset_scale, corr_alpha_max)
        self.main1 = GeneratedDiagonalLinear(d_model, dim_ff, rank_main, num_tasks, task_dim, d_model, tune_scale, name=f"{name}_main1")
        self.main2 = GeneratedDiagonalLinear(dim_ff, d_model, rank_main, num_tasks, task_dim, dim_ff, tune_scale, name=f"{name}_main2")
        self.corr1 = GeneratedDiagonalLinear(d_model, dim_ff, rank_corr, num_tasks, task_dim, d_model, tune_scale, name=f"{name}_corr1") if correction_mode in {"input", "both"} and rank_corr > 0 else None
        self.corr2 = GeneratedDiagonalLinear(dim_ff, d_model, rank_corr, num_tasks, task_dim, dim_ff, tune_scale, name=f"{name}_corr2") if correction_mode in {"output", "both"} and rank_corr > 0 else None
        self.last: Dict[str, float] = {}
        self.last_budget: Optional[torch.Tensor] = None
        self.last_binary: Optional[torch.Tensor] = None
        self.last_ratio_penalty: Optional[torch.Tensor] = None
        self.last_task_offset_l2: Optional[torch.Tensor] = None

    @staticmethod
    def _targeted(raw: torch.Tensor, base: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        raw_norm = raw.norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True).clamp_min(eps)
        base_norm = base.detach().norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True).clamp_min(eps)
        return raw / raw_norm * base_norm * target[:, None, :]

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        ctx = x.mean(dim=1)
        te = self.task_emb(task_ids)
        gate, alpha, target1, target2, cgate, calpha, raw_router = self.router(ctx, te, task_ids)

        base1 = self.base_l1(x)
        raw_main1 = self.main1(x, task_ids, ctx, te)
        main1 = gate[:, None, :] * alpha[:, None, :] * self._targeted(raw_main1, base1, target1)
        corr1 = torch.zeros_like(base1)
        if self.corr1 is not None:
            corr1 = cgate[:, None, :] * calpha[:, None, :] * self.corr_gain * self.corr1(x, task_ids, ctx, te)
        h = F.gelu(base1 + main1 + corr1)

        ctx2 = h.mean(dim=1)
        base2 = self.base_l2(h)
        raw_main2 = self.main2(h, task_ids, ctx2, te)
        main2 = gate[:, None, :] * alpha[:, None, :] * self._targeted(raw_main2, base2, target2)
        corr2 = torch.zeros_like(base2)
        if self.corr2 is not None:
            corr2 = cgate[:, None, :] * calpha[:, None, :] * self.corr_gain * self.corr2(h, task_ids, ctx2, te)

        y = base2 + main2 + corr2
        self._record(task_ids, base1, base2, main1, main2, corr1, corr2, gate, alpha, cgate, calpha, target1, target2, raw_router)
        return y

    def _record(self, task_ids, base1, base2, main1, main2, corr1, corr2, gate, alpha, cgate, calpha, target1, target2, raw_router) -> None:
        eps = 1e-6
        ratio1 = (main1 + corr1).norm(dim=-1).mean() / (base1.detach().norm(dim=-1).mean() + eps)
        ratio2 = (main2 + corr2).norm(dim=-1).mean() / (base2.detach().norm(dim=-1).mean() + eps)
        corr_ratio = 0.5 * (corr1.norm(dim=-1).mean() / (base1.detach().norm(dim=-1).mean() + eps) + corr2.norm(dim=-1).mean() / (base2.detach().norm(dim=-1).mean() + eps))
        total_ratio = 0.5 * (ratio1 + ratio2)
        budget = 0.5 * gate.mean() * (self.main1.budget_tensor(task_ids) + self.main2.budget_tensor(task_ids))
        if self.corr1 is not None:
            budget = budget + 0.25 * cgate.mean() * self.corr1.budget_tensor(task_ids)
        if self.corr2 is not None:
            budget = budget + 0.25 * cgate.mean() * self.corr2.budget_tensor(task_ids)
        binary = (gate * (1.0 - gate)).mean() + 0.5 * (cgate * (1.0 - cgate)).mean()
        ratio_penalty = F.relu(total_ratio - self.ratio_ceiling).pow(2) + 0.5 * F.relu(corr_ratio - self.corr_ceiling).pow(2)
        offset_l2 = self.router.task_offset.weight.pow(2).mean()
        self.last_budget = budget
        self.last_binary = binary
        self.last_ratio_penalty = ratio_penalty
        self.last_task_offset_l2 = offset_l2
        self.last = {
            f"{self.name}_gate_main": float(gate.detach().mean()),
            f"{self.name}_alpha_main": float(alpha.detach().mean()),
            f"{self.name}_gate_corr": float(cgate.detach().mean()),
            f"{self.name}_alpha_corr": float(calpha.detach().mean()),
            f"{self.name}_target1": float(target1.detach().mean()),
            f"{self.name}_target2": float(target2.detach().mean()),
            f"{self.name}_delta_base_ratio": float(total_ratio.detach()),
            f"{self.name}_corr_ratio": float(corr_ratio.detach()),
            f"{self.name}_budget": float(budget.detach()),
            f"{self.name}_binary": float(binary.detach()),
            f"{self.name}_ratio_penalty": float(ratio_penalty.detach()),
            f"{self.name}_task_offset_l2": float(offset_l2.detach()),
        }
        for module in [self.main1, self.main2, self.corr1, self.corr2]:
            if module is not None:
                self.last.update(module.diagnostics())

    def hts_losses(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = next(self.parameters()).device
        zero = torch.tensor(0.0, device=dev)
        return (
            self.last_budget if self.last_budget is not None else zero,
            self.last_binary if self.last_binary is not None else zero,
            self.last_ratio_penalty if self.last_ratio_penalty is not None else zero,
            self.last_task_offset_l2 if self.last_task_offset_l2 is not None else zero,
        )

    def diagnostics(self) -> Dict[str, float]:
        return dict(self.last)


class StaticFFN(nn.Module):
    def __init__(self, d_model: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model))

    def forward(self, x: torch.Tensor, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x)

    def hts_losses(self):
        dev = next(self.parameters()).device
        z = torch.tensor(0.0, device=dev)
        return z, z, z, z

    def diagnostics(self) -> Dict[str, float]:
        return {}
