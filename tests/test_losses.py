import torch
import pytest
from hts.losses import cross_entropy_with_margin, hts_regularization_loss
from hts.config import HtSConfig
from hts.models import HtSTransformerClassifier, StaticTransformerClassifier
from hts.data.synthetic_tasks import SyntheticTaskBatcher


class TestCrossEntropyWithMargin:
    def test_basic_forward(self):
        logits = torch.randn(8, 128)
        labels = torch.randint(0, 128, (8,))
        total_loss, ce, margin_loss = cross_entropy_with_margin(logits, labels)
        assert total_loss.shape == ()
        assert ce.shape == ()
        assert margin_loss.shape == ()

    def test_total_loss_greater_than_ce(self):
        logits = torch.randn(8, 128)
        labels = torch.randint(0, 128, (8,))
        total_loss, ce, margin_loss = cross_entropy_with_margin(logits, labels, margin_weight=0.1)
        assert total_loss.item() >= ce.item() - 1e-6

    def test_zero_margin_weight(self):
        logits = torch.randn(8, 128)
        labels = torch.randint(0, 128, (8,))
        total_loss, ce, margin_loss = cross_entropy_with_margin(logits, labels, margin_weight=0.0)
        assert torch.allclose(total_loss, ce)
        assert margin_loss.item() == 0.0

    def test_perfect_predictions(self):
        logits = torch.eye(8, 128) * 100
        labels = torch.arange(8)
        total_loss, ce, margin_loss = cross_entropy_with_margin(logits, labels, margin_weight=0.1)
        assert margin_loss.item() == 0.0

    def test_wrong_predictions(self):
        logits = torch.zeros(8, 128)
        logits[:, 1] = 100
        labels = torch.zeros(8, dtype=torch.long)
        total_loss, ce, margin_loss = cross_entropy_with_margin(logits, labels, margin_weight=0.1)
        assert margin_loss.item() > 0.0

    def test_margin_parameter(self):
        logits = torch.randn(8, 128)
        labels = torch.randint(0, 128, (8,))
        _, _, ml_small = cross_entropy_with_margin(logits, labels, margin=0.1)
        _, _, ml_large = cross_entropy_with_margin(logits, labels, margin=1.0)
        assert ml_large.item() >= ml_small.item() - 1e-6


class TestHtSRegularizationLoss:
    def test_hts_model(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        model(batch["input_ids"], batch["task_ids"])
        reg, parts = hts_regularization_loss(model)
        assert reg.shape == ()
        assert "reg_total" in parts
        assert "budget_loss" in parts
        assert "binary_loss" in parts
        assert "ratio_loss" in parts
        assert "task_offset_l2" in parts

    def test_static_model(self):
        cfg = TransformerConfig(d_model=24, n_heads=4, dim_ff=32)
        model = StaticTransformerClassifier(cfg)
        reg, parts = hts_regularization_loss(model)
        assert reg.item() == 0.0
        assert parts["budget_loss"] == 0.0

    def test_custom_weights(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        model(batch["input_ids"], batch["task_ids"])
        reg1, _ = hts_regularization_loss(model, budget_weight=0.1)
        reg2, _ = hts_regularization_loss(model, budget_weight=0.0)
        assert reg1.item() >= reg2.item() - 1e-6


from hts.config import TransformerConfig
