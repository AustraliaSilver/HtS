"""Tests for compositional task arithmetic benchmark."""
import pytest
import torch
from hts.data.compositional_tasks import (
    CompositionalBatcher, COMPOSITIONS, OP_NAMES, BASE_OPS,
    COMP_ID_OFFSET, _tok, TOKEN_THEN, NUM_OFFSET, PAD, CLS, SEP,
)


class TestBaseOps:
    def test_add(self):
        assert BASE_OPS["add"](3, 5) == 8

    def test_sub(self):
        assert BASE_OPS["sub"](5, 3) == 2
        assert BASE_OPS["sub"](2, 5) == 0  # clamped

    def test_mul(self):
        assert BASE_OPS["mul"](3, 4) == 12

    def test_max(self):
        assert BASE_OPS["max"](3, 7) == 7

    def test_min(self):
        assert BASE_OPS["min"](3, 7) == 3

    def test_mod(self):
        assert BASE_OPS["mod"](7, 3) == 1

    def test_gt(self):
        assert BASE_OPS["gt"](5, 3) == 1
        assert BASE_OPS["gt"](2, 5) == 0

    def test_parity(self):
        assert BASE_OPS["parity"](3, 5) == 0  # 8 % 2
        assert BASE_OPS["parity"](3, 4) == 1  # 7 % 2


class TestCompositions:
    def test_all_pairs_exist(self):
        # 8 ops × 7 other ops = 56 compositions
        assert len(COMPOSITIONS) == 56

    def test_no_self_composition(self):
        for comp in COMPOSITIONS:
            assert comp["op1"] != comp["op2"]

    def test_unique_ids(self):
        ids = [c["id"] for c in COMPOSITIONS]
        assert len(ids) == len(set(ids))

    def test_compose_add_mul(self):
        comp = next(c for c in COMPOSITIONS if c["op1"] == "add" and c["op2"] == "mul")
        # COMPOSE(add, mul)(2, 3, 4) = mul(add(2,3), 4) = mul(5, 4) = 20
        result = comp["fn2"](comp["fn1"](2, 3), 4)
        assert result == 20

    def test_compose_max_sub(self):
        comp = next(c for c in COMPOSITIONS if c["op1"] == "max" and c["op2"] == "sub")
        # COMPOSE(max, sub)(3, 7, 2) = sub(max(3,7), 2) = sub(7, 2) = 5
        result = comp["fn2"](comp["fn1"](3, 7), 2)
        assert result == 5


class TestCompositionalBatcher:
    def test_init(self):
        batcher = CompositionalBatcher()
        assert batcher.max_num == 10
        assert batcher.output_dim == 128

    def test_sample_one(self):
        batcher = CompositionalBatcher()
        tokens, task_id, label = batcher.sample_one()
        assert isinstance(tokens, list)
        assert len(tokens) == batcher.max_len
        assert tokens[0] == CLS
        assert tokens[4] == TOKEN_THEN
        assert task_id >= COMP_ID_OFFSET
        assert 0 <= label < batcher.output_dim

    def test_batch(self):
        batcher = CompositionalBatcher()
        batch = batcher.batch(16)
        assert batch["input_ids"].shape == (16, batcher.max_len)
        assert batch["task_ids"].shape == (16,)
        assert batch["labels"].shape == (16,)

    def test_batch_device(self):
        batcher = CompositionalBatcher()
        batch = batcher.batch(8, device="cpu")
        assert batch["input_ids"].device.type == "cpu"

    def test_num_tasks(self):
        batcher = CompositionalBatcher()
        assert batcher.num_tasks == COMP_ID_OFFSET + len(COMPOSITIONS)

    def test_task_id_range(self):
        batcher = CompositionalBatcher()
        batch = batcher.batch(64)
        task_ids = batch["task_ids"].unique()
        for tid in task_ids:
            assert tid.item() >= COMP_ID_OFFSET
            assert tid.item() < COMP_ID_OFFSET + len(COMPOSITIONS)

    def test_label_range(self):
        batcher = CompositionalBatcher()
        batch = batcher.batch(64)
        assert batch["labels"].min() >= 0
        assert batch["labels"].max() < 128

    def test_token_format(self):
        batcher = CompositionalBatcher()
        tokens, _, _ = batcher.sample_one()
        # [CLS] a b c THEN op1 op2 [SEP] PAD...
        assert tokens[0] == CLS
        assert tokens[4] == TOKEN_THEN
        # After THEN, two op tokens should be in range [NUM_OFFSET, NUM_OFFSET+8)
        assert NUM_OFFSET <= tokens[5] < NUM_OFFSET + 8
        assert NUM_OFFSET <= tokens[6] < NUM_OFFSET + 8
