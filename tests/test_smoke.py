import torch

from hts_b12 import (
    HtSB12Classifier,
    HtSB12Config,
    LabelSpec,
    ModelGroupConfig,
    ModelGroupRegistry,
    TaskSpec,
    build_hts_config_from_group,
)
from hts_b12.groups import HtSBatch


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
