import torch
from torch import nn
import pytest

from hts_b12 import (
    HtSB12Classifier,
    HtSB12Config,
    LabelSpec,
    ModelGroupConfig,
    ModelGroupRegistry,
    TaskSpec,
    build_hts_config_from_group,
    TrainConfig,
    TransformerClassifier,
)
from hts_b12.groups import HtSBatch
from hts_b12.training import train_classifier


def test_forward_shape():
    cfg = HtSB12Config(vocab_size=32, num_tasks=2, num_classes=5, max_length=8, d_model=32, n_heads=4, num_layers=1, dim_ff=64, task_dim=8, rank_main=2, rank_corr=1)
    model = HtSB12Classifier(cfg)
    x = torch.randint(0, 32, (4, 8))
    task = torch.randint(0, 2, (4,))
    logits = model(x, task)
    assert logits.shape == (4, 5)


def test_group_config_builds_model():
    group = ModelGroupConfig(
        name="toy",
        vocab_size=20,
        max_length=10,
        tasks=[TaskSpec("a", 0), TaskSpec("b", 1)],
        labels=LabelSpec(num_classes=3),
        recommended_model={"d_model": 32, "n_heads": 4, "num_layers": 1, "dim_ff": 64, "task_dim": 8, "rank_main": 2, "rank_corr": 1},
    )
    cfg = build_hts_config_from_group(group)
    assert cfg.vocab_size == 20
    assert cfg.num_tasks == 2
    assert cfg.num_classes == 3
    assert HtSB12Classifier(cfg)(torch.randint(0, 20, (2, 10)), torch.zeros(2, dtype=torch.long)).shape == (2, 3)


def test_registry_batch_factory():
    reg = ModelGroupRegistry()
    group = ModelGroupConfig(name="toy", vocab_size=10, max_length=4, tasks=[TaskSpec("task", 0)], labels=LabelSpec(num_classes=2))

    def factory(batch_size, device, seed):
        return HtSBatch(
            input_ids=torch.ones(batch_size, 4, dtype=torch.long, device=device),
            task_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            labels=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    reg.register(group, factory)
    batch = reg.factory("toy")(3, torch.device("cpu"), 1)
    assert batch.input_ids.shape == (3, 4)
    assert batch.labels.tolist() == [0, 0, 0]


def test_config_rejects_invalid_attention_heads():
    with pytest.raises(ValueError, match="d_model must be divisible"):
        HtSB12Config(d_model=30, n_heads=8)


def test_transformer_mean_pool_ignores_padding_tokens():
    cfg = HtSB12Config(
        vocab_size=16,
        num_tasks=1,
        num_classes=3,
        max_length=4,
        d_model=16,
        n_heads=4,
        num_layers=1,
        dim_ff=32,
        dropout=0.0,
        pool="mean",
        use_cls_token=False,
    )
    model = TransformerClassifier(cfg).eval()
    task_ids = torch.zeros(2, dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.long)
    input_ids = torch.tensor([[1, 2, 0, 0], [1, 2, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, task_ids, attention_mask)

    assert torch.allclose(logits[0], logits[1], atol=1e-5)


def test_train_classifier_selects_best_on_eval_batch():
    class EchoTokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, input_ids, task_ids, attention_mask=None):
            logits = torch.zeros(input_ids.size(0), 2, device=input_ids.device)
            logits.scatter_(1, input_ids[:, :1], 1.0)
            return logits * self.scale

    def train_batch(batch_size, device, seed):
        return HtSBatch(
            input_ids=torch.zeros(batch_size, 1, dtype=torch.long, device=device),
            task_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            labels=torch.ones(batch_size, dtype=torch.long, device=device),
        )

    def eval_batch(batch_size, device, seed):
        return HtSBatch(
            input_ids=torch.ones(batch_size, 1, dtype=torch.long, device=device),
            task_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            labels=torch.ones(batch_size, dtype=torch.long, device=device),
        )

    log = train_classifier(
        EchoTokenModel(),
        train_batch,
        TrainConfig(steps=1, batch_size=4, eval_every=1, device="cpu", save_best=False),
        eval_batch_fn=eval_batch,
    )

    assert log.best_acc == 1.0
    assert log.rows[0]["accuracy"] == 1.0
    assert log.rows[0]["train_accuracy"] == 0.0
