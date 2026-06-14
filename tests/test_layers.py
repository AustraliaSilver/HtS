import torch
import pytest
from hts.layers import GeneratedDiagonalLinear, RatioRouter, HtSB12FFN, StaticFFN


class TestGeneratedDiagonalLinear:
    def test_output_shape(self):
        B, L, In, Out, R = 4, 10, 32, 64, 5
        layer = GeneratedDiagonalLinear(In, Out, R, num_tasks=10, task_dim=8, ctx_dim=In)
        x = torch.randn(B, L, In)
        task_ids = torch.randint(0, 10, (B,))
        ctx = x.mean(dim=1)
        task_emb = torch.randn(B, 8)
        out = layer(x, task_ids, ctx, task_emb)
        assert out.shape == (B, L, Out)

    def test_coefficients_shape(self):
        B, R = 8, 5
        layer = GeneratedDiagonalLinear(32, 64, R, num_tasks=10, task_dim=8, ctx_dim=32)
        task_ids = torch.randint(0, 10, (B,))
        ctx = torch.randn(B, 32)
        task_emb = torch.randn(B, 8)
        coeff = layer.coefficients(task_ids, ctx, task_emb)
        assert coeff.shape == (B, R)

    def test_coefficients_range(self):
        layer = GeneratedDiagonalLinear(32, 64, 5, num_tasks=10, task_dim=8, ctx_dim=32)
        task_ids = torch.randint(0, 10, (8,))
        ctx = torch.randn(8, 32)
        task_emb = torch.randn(8, 8)
        coeff = layer.coefficients(task_ids, ctx, task_emb)
        assert coeff.min().item() >= -1.0
        assert coeff.max().item() <= 1.0

    def test_budget_tensor(self):
        layer = GeneratedDiagonalLinear(32, 64, 5, num_tasks=10, task_dim=8, ctx_dim=32)
        task_ids = torch.randint(0, 10, (8,))
        budget = layer.budget_tensor(task_ids)
        assert budget.shape == ()
        assert 0.0 <= budget.item() <= 1.0

    def test_diagnostics(self):
        layer = GeneratedDiagonalLinear(32, 64, 5, num_tasks=10, task_dim=8, ctx_dim=32)
        task_ids = torch.randint(0, 10, (8,))
        ctx = torch.randn(8, 32)
        task_emb = torch.randn(8, 8)
        layer.coefficients(task_ids, ctx, task_emb)
        diag = layer.diagnostics()
        assert "gen_coeff_abs" in diag
        assert "gen_rank_eff" in diag
        assert "gen_mask_mean" in diag

    def test_backward(self):
        layer = GeneratedDiagonalLinear(32, 64, 5, num_tasks=10, task_dim=8, ctx_dim=32)
        x = torch.randn(4, 10, 32, requires_grad=True)
        task_ids = torch.randint(0, 10, (4,))
        ctx = x.mean(dim=1)
        task_emb = torch.randn(4, 8)
        out = layer(x, task_ids, ctx, task_emb)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestRatioRouter:
    def test_output_shapes(self):
        router = RatioRouter(ctx_dim=32, task_dim=8, num_tasks=10, alpha_max=1.18, target_min=0.34, target_max=0.90)
        ctx = torch.randn(4, 32)
        task_emb = torch.randn(4, 8)
        task_ids = torch.randint(0, 10, (4,))
        gate, alpha, target1, target2, cgate, calpha, raw = router(ctx, task_emb, task_ids)
        B = 4
        assert gate.shape == (B, 1)
        assert alpha.shape == (B, 1)
        assert target1.shape == (B, 1)
        assert target2.shape == (B, 1)
        assert cgate.shape == (B, 1)
        assert calpha.shape == (B, 1)
        assert raw.shape == (B, 6)

    def test_gate_range(self):
        router = RatioRouter(ctx_dim=32, task_dim=8, num_tasks=10, alpha_max=1.18, target_min=0.34, target_max=0.90)
        ctx = torch.randn(8, 32)
        task_emb = torch.randn(8, 8)
        task_ids = torch.randint(0, 10, (8,))
        gate, alpha, _, _, cgate, calpha, _ = router(ctx, task_emb, task_ids)
        assert gate.min().item() >= 0.0
        assert gate.max().item() <= 1.0
        assert alpha.min().item() >= 0.0
        assert alpha.max().item() <= 1.18
        assert cgate.min().item() >= 0.0
        assert cgate.max().item() <= 1.0
        assert calpha.min().item() >= 0.0
        assert calpha.max().item() <= 0.55

    def test_target_range(self):
        router = RatioRouter(ctx_dim=32, task_dim=8, num_tasks=10, alpha_max=1.18, target_min=0.34, target_max=0.90)
        ctx = torch.randn(8, 32)
        task_emb = torch.randn(8, 8)
        task_ids = torch.randint(0, 10, (8,))
        _, _, target1, target2, _, _, _ = router(ctx, task_emb, task_ids)
        assert target1.min().item() >= 0.34
        assert target1.max().item() <= 0.90
        assert target2.min().item() >= 0.34
        assert target2.max().item() <= 0.90


class TestHtSB12FFN:
    def test_output_shape(self):
        ffn = HtSB12FFN(d_model=32, dim_ff=64, num_tasks=10, task_dim=8, rank_main=5, rank_corr=2)
        x = torch.randn(4, 10, 32)
        task_ids = torch.randint(0, 10, (4,))
        out = ffn(x, task_ids)
        assert out.shape == (4, 10, 32)

    def test_hts_losses(self):
        ffn = HtSB12FFN(d_model=32, dim_ff=64, num_tasks=10, task_dim=8)
        x = torch.randn(4, 10, 32)
        task_ids = torch.randint(0, 10, (4,))
        ffn(x, task_ids)
        budget, binary, ratio, offset = ffn.hts_losses()
        assert budget.shape == ()
        assert binary.shape == ()
        assert ratio.shape == ()
        assert offset.shape == ()

    def test_diagnostics(self):
        ffn = HtSB12FFN(d_model=32, dim_ff=64, num_tasks=10, task_dim=8)
        x = torch.randn(4, 10, 32)
        task_ids = torch.randint(0, 10, (4,))
        ffn(x, task_ids)
        diag = ffn.diagnostics()
        assert "hts_ffn_gate_main" in diag
        assert "hts_ffn_alpha_main" in diag
        assert "hts_ffn_target1" in diag
        assert "hts_ffn_target2" in diag
        assert "hts_ffn_delta_base_ratio" in diag
        assert "hts_ffn_budget" in diag

    def test_correction_modes(self):
        for mode in ["input", "output", "both", "none"]:
            ffn = HtSB12FFN(d_model=32, dim_ff=64, num_tasks=10, task_dim=8, correction_mode=mode)
            x = torch.randn(4, 10, 32)
            task_ids = torch.randint(0, 10, (4,))
            out = ffn(x, task_ids)
            assert out.shape == (4, 10, 32)

    def test_backward(self):
        ffn = HtSB12FFN(d_model=32, dim_ff=64, num_tasks=10, task_dim=8)
        x = torch.randn(4, 10, 32, requires_grad=True)
        task_ids = torch.randint(0, 10, (4,))
        out = ffn(x, task_ids)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestStaticFFN:
    def test_output_shape(self):
        ffn = StaticFFN(d_model=32, dim_ff=64)
        x = torch.randn(4, 10, 32)
        out = ffn(x)
        assert out.shape == (4, 10, 32)

    def test_hts_losses_returns_zeros(self):
        ffn = StaticFFN(d_model=32, dim_ff=64)
        budget, binary, ratio, offset = ffn.hts_losses()
        assert budget.item() == 0.0
        assert binary.item() == 0.0
        assert ratio.item() == 0.0
        assert offset.item() == 0.0

    def test_diagnostics_empty(self):
        ffn = StaticFFN(d_model=32, dim_ff=64)
        diag = ffn.diagnostics()
        assert diag == {}

    def test_with_dropout(self):
        ffn = StaticFFN(d_model=32, dim_ff=64, dropout=0.1)
        x = torch.randn(4, 10, 32)
        out = ffn(x)
        assert out.shape == (4, 10, 32)
