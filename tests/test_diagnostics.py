import torch
import pytest
from hts.diagnostics import accuracy, collect_diagnostics, summarize_batch
from hts.config import HtSConfig, TransformerConfig
from hts.models import HtSTransformerClassifier, StaticTransformerClassifier
from hts.data.synthetic_tasks import SyntheticTaskBatcher


class TestAccuracy:
    def test_perfect_accuracy(self):
        logits = torch.eye(8, 128) * 100
        labels = torch.arange(8)
        acc = accuracy(logits, labels)
        assert acc == 1.0

    def test_zero_accuracy(self):
        logits = torch.zeros(8, 128)
        logits[:, 1] = 100
        labels = torch.zeros(8, dtype=torch.long)
        acc = accuracy(logits, labels)
        assert acc == 0.0

    def test_random_accuracy(self):
        logits = torch.randn(100, 128)
        labels = torch.randint(0, 128, (100,))
        acc = accuracy(logits, labels)
        assert 0.0 <= acc <= 1.0


class TestCollectDiagnostics:
    def test_hts_model(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(4, family="arith8")
        model(batch["input_ids"], batch["task_ids"])
        diag = collect_diagnostics(model)
        assert len(diag) > 0

    def test_static_model(self):
        cfg = TransformerConfig(d_model=24, n_heads=4, dim_ff=32)
        model = StaticTransformerClassifier(cfg)
        diag = collect_diagnostics(model)
        assert diag == {}

    def test_no_diagnostics_method(self):
        obj = object()
        diag = collect_diagnostics(obj)
        assert diag == {}


class TestSummarizeBatch:
    def test_perfect_confidence(self):
        logits = torch.eye(8, 128) * 100
        labels = torch.arange(8)
        summary = summarize_batch(logits, labels)
        assert summary["accuracy"] == 1.0
        assert summary["mean_confidence"] == 1.0
        assert summary["mean_true_prob"] == 1.0

    def test_output_keys(self):
        logits = torch.randn(8, 128)
        labels = torch.randint(0, 128, (8,))
        summary = summarize_batch(logits, labels)
        assert "accuracy" in summary
        assert "mean_confidence" in summary
        assert "mean_true_prob" in summary

    def test_confidence_range(self):
        logits = torch.randn(32, 128)
        labels = torch.randint(0, 128, (32,))
        summary = summarize_batch(logits, labels)
        assert 0.0 <= summary["mean_confidence"] <= 1.0
        assert 0.0 <= summary["mean_true_prob"] <= 1.0
