import torch
from hts import HtSConfig, TransformerConfig
from hts.models import HtSTransformerClassifier, StaticTransformerClassifier
from hts.data.synthetic_tasks import SyntheticTaskBatcher, OUTPUT_DIM
from hts.device import detect_device


def test_device_detection():
    info = detect_device("cpu")
    assert info.backend == "cpu"


def test_hts_forward_backward_cpu():
    cfg = HtSConfig(d_model=24, n_heads=4, dim_ff=32, rank_main=3, rank_corr=1)
    model = HtSTransformerClassifier(cfg)
    batch = SyntheticTaskBatcher().batch(8, family="arith8")
    logits = model(batch["input_ids"], batch["task_ids"])
    assert logits.shape == (8, OUTPUT_DIM)
    loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert model.diagnostics()


def test_transformer_forward_cpu():
    cfg = TransformerConfig(d_model=24, n_heads=4, dim_ff=32)
    model = StaticTransformerClassifier(cfg)
    batch = SyntheticTaskBatcher().batch(8, family="seq6")
    logits = model(batch["input_ids"], batch["task_ids"])
    assert logits.shape == (8, OUTPUT_DIM)
