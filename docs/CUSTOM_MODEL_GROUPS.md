# Custom Model Groups in HtS-B12

HtS-B12 is designed around **model groups**. A group is the boundary where you define:

- vocabulary/token space;
- maximum sequence length;
- tasks and task ids;
- label space;
- recommended model hyperparameters;
- a batch factory or dataset adapter.

## Minimal Group

```python
from hts_b12 import ModelGroupConfig, TaskSpec, LabelSpec

group = ModelGroupConfig(
    name="my_group",
    vocab_size=1000,
    max_length=128,
    tasks=[TaskSpec("task_a", 0), TaskSpec("task_b", 1)],
    labels=LabelSpec(num_classes=10),
)
```

## Batch Adapter Contract

A batch factory must have this signature:

```python
def batch_factory(batch_size: int, device: torch.device, seed: int) -> HtSBatch:
    ...
```

and return:

```python
HtSBatch(input_ids, task_ids, labels, attention_mask=None)
```

## Why This Design?

HtS-B12 uses `task_ids` as the hard condition for soft-weight generation. Model groups make that explicit and avoid hiding task logic inside the benchmark script.

## Multi-Group Training

Use `make_multi_group_factory` to mix several registered groups:

```python
from hts_b12 import make_multi_group_factory

factory = make_multi_group_factory(registry, ["group_a", "group_b"])
```

This uniformly samples one group per batch.
