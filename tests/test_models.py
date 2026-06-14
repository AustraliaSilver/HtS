import torch
import pytest
from hts.config import HtSConfig, TransformerConfig
from hts.models import (
    HtSTransformerClassifier,
    StaticTransformerClassifier,
    TokenPosEmbedding,
    TransformerBlock,
    count_parameters,
)
from hts.data.synthetic_tasks import SyntheticTaskBatcher, VOCAB_SIZE, OUTPUT_DIM, MAX_LEN


class TestTokenPosEmbedding:
    def test_output_shape(self):
        emb = TokenPosEmbedding(vocab_size=64, max_len=12, d_model=32)
        x = torch.randint(0, 64, (4, 10))
        out = emb(x)
        assert out.shape == (4, 10, 32)

    def test_max_len_exceeded(self):
        emb = TokenPosEmbedding(vocab_size=64, max_len=5, d_model=32)
        x = torch.randint(0, 64, (4, 10))
        with pytest.raises(ValueError, match="Sequence length"):
            emb(x)


class TestTransformerBlock:
    def test_hts_losses_without_hts(self):
        from hts.layers import StaticFFN
        block = TransformerBlock(d_model=32, n_heads=4, ffn=StaticFFN(32, 64))
        budget, binary, ratio, offset = block.hts_losses()
        assert budget.item() == 0.0

    def test_diagnostics_without_hts(self):
        from hts.layers import StaticFFN
        block = TransformerBlock(d_model=32, n_heads=4, ffn=StaticFFN(32, 64))
        diag = block.diagnostics()
        assert diag == {}


class TestHtSTransformerClassifier:
    def test_output_shape(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        logits = model(batch["input_ids"], batch["task_ids"])
        assert logits.shape == (8, OUTPUT_DIM)

    def test_diagnostics(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        model(batch["input_ids"], batch["task_ids"])
        diag = model.diagnostics()
        assert len(diag) > 0
        assert "block0.hts_l0_gate_main" in diag

    def test_hts_losses(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        model(batch["input_ids"], batch["task_ids"])
        budget, binary, ratio, offset = model.hts_losses()
        assert budget.shape == ()

    def test_multiple_layers(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, n_layers=2, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        logits = model(batch["input_ids"], batch["task_ids"])
        assert logits.shape == (8, OUTPUT_DIM)
        diag = model.diagnostics()
        assert "block0.hts_l0_gate_main" in diag
        assert "block1.hts_l1_gate_main" in diag

    def test_backward(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        logits = model(batch["input_ids"], batch["task_ids"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_count_parameters(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        params = count_parameters(model)
        assert params > 0


class TestStaticTransformerClassifier:
    def test_output_shape(self):
        cfg = TransformerConfig(d_model=24, n_heads=4, dim_ff=32)
        model = StaticTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        logits = model(batch["input_ids"], batch["task_ids"])
        assert logits.shape == (8, OUTPUT_DIM)

    def test_diagnostics_empty(self):
        cfg = TransformerConfig(d_model=24, n_heads=4, dim_ff=32)
        model = StaticTransformerClassifier(cfg)
        diag = model.diagnostics()
        assert diag == {}

    def test_hts_losses_zeros(self):
        cfg = TransformerConfig(d_model=24, n_heads=4, dim_ff=32)
        model = StaticTransformerClassifier(cfg)
        budget, binary, ratio, offset = model.hts_losses()
        assert budget.item() == 0.0

    def test_backward(self):
        cfg = TransformerConfig(d_model=24, n_heads=4, dim_ff=32)
        model = StaticTransformerClassifier(cfg)
        batch = SyntheticTaskBatcher().batch(8, family="arith8")
        logits = model(batch["input_ids"], batch["task_ids"])
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


class TestCountParameters:
    def test_counts_correctly(self):
        cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
        model = HtSTransformerClassifier(cfg)
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count_parameters(model) == expected
