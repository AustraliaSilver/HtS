import pytest
from hts.config import HtSConfig, TransformerConfig, TrainConfig


class TestHtSConfig:
    def test_default_values(self):
        cfg = HtSConfig()
        assert cfg.vocab_size == 64
        assert cfg.output_dim == 128
        assert cfg.num_tasks == 30
        assert cfg.max_len == 12
        assert cfg.d_model == 40
        assert cfg.n_heads == 4
        assert cfg.dim_ff == 64
        assert cfg.n_layers == 1
        assert cfg.task_dim == 16
        assert cfg.rank_main == 5
        assert cfg.rank_corr == 2
        assert cfg.alpha_max == 1.18
        assert cfg.target_min == 0.34
        assert cfg.target_max == 0.90
        assert cfg.tune_scale == 0.34
        assert cfg.gate_bias == -0.06
        assert cfg.task_offset_scale == 0.30
        assert cfg.corr_alpha_max == 0.55
        assert cfg.corr_gain == 0.55
        assert cfg.ratio_ceiling == 1.35
        assert cfg.corr_ceiling == 0.55
        assert cfg.correction_mode == "input"
        assert cfg.dropout == 0.0

    def test_custom_values(self):
        cfg = HtSConfig(d_model=64, dim_ff=128, n_layers=2, rank_main=8)
        assert cfg.d_model == 64
        assert cfg.dim_ff == 128
        assert cfg.n_layers == 2
        assert cfg.rank_main == 8

    def test_dataclass_fields(self):
        cfg = HtSConfig()
        fields = {f.name for f in HtSConfig.__dataclass_fields__.values()}
        assert "vocab_size" in fields
        assert "d_model" in fields
        assert "n_layers" in fields


class TestTransformerConfig:
    def test_default_values(self):
        cfg = TransformerConfig()
        assert cfg.vocab_size == 64
        assert cfg.output_dim == 128
        assert cfg.num_tasks == 30
        assert cfg.max_len == 12
        assert cfg.d_model == 40
        assert cfg.n_heads == 4
        assert cfg.dim_ff == 64
        assert cfg.n_layers == 1
        assert cfg.dropout == 0.0

    def test_custom_values(self):
        cfg = TransformerConfig(d_model=128, n_layers=3, dropout=0.1)
        assert cfg.d_model == 128
        assert cfg.n_layers == 3
        assert cfg.dropout == 0.1


class TestTrainConfig:
    def test_default_values(self):
        cfg = TrainConfig()
        assert cfg.steps == 300
        assert cfg.batch_size == 64
        assert cfg.lr == 3e-3
        assert cfg.eval_every == 50
        assert cfg.eval_batches == 10
        assert cfg.seed == 42
        assert cfg.device == "auto"
        assert cfg.margin_weight == 0.05
        assert cfg.margin == 0.35
        assert cfg.budget_weight == 1e-4
        assert cfg.binary_weight == 1e-4
        assert cfg.ratio_weight == 5e-4
        assert cfg.task_offset_weight == 1e-5
        assert cfg.grad_clip == 1.0

    def test_custom_values(self):
        cfg = TrainConfig(steps=1000, batch_size=128, lr=1e-3, device="cuda")
        assert cfg.steps == 1000
        assert cfg.batch_size == 128
        assert cfg.lr == 1e-3
        assert cfg.device == "cuda"
