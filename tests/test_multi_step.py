"""Tests for multi-step reasoning chain benchmark."""
import pytest
import torch
from hts.data.multi_step_reasoning import (
    MultiStepBatcher, CHAIN_TEMPLATES, CHAIN_ID_OFFSET,
    _compute_chain, _tok, PAD, CLS, SEP, TOKEN_OP, MAX_CHAIN_LEN, MAX_TOTAL_LEN,
)


class TestComputeChain:
    def test_add_mul(self):
        tmpl = CHAIN_TEMPLATES[0]  # ADD_MUL
        result = _compute_chain(tmpl, [3, 5, 2])
        assert result == (3 + 5) * 2  # 16

    def test_max_sub(self):
        tmpl = CHAIN_TEMPLATES[1]  # MAX_SUB
        result = _compute_chain(tmpl, [3, 7, 2])
        assert result == 7 - 2  # 5

    def test_add_add_mod(self):
        tmpl = CHAIN_TEMPLATES[2]  # ADD_ADD_MOD
        result = _compute_chain(tmpl, [1, 2, 3, 5])
        assert result == (1 + 2 + 3) % 5  # 1

    def test_ifgt_add_true(self):
        tmpl = CHAIN_TEMPLATES[3]  # IFGT_ADD
        result = _compute_chain(tmpl, [5, 3, 10, 2])
        assert result == 10 + 2  # 5 > 3 → True

    def test_ifgt_add_false(self):
        tmpl = CHAIN_TEMPLATES[3]  # IFGT_ADD
        result = _compute_chain(tmpl, [2, 5, 10, 2])
        assert result == 10 - 2  # 2 > 5 → False

    def test_mul_mod_add(self):
        tmpl = CHAIN_TEMPLATES[4]  # MUL_MOD_ADD
        result = _compute_chain(tmpl, [3, 4, 7, 2])
        assert result == ((3 * 4) % 7) + 2  # 12%7 + 2 = 5+2 = 7

    def test_sub_mul(self):
        tmpl = CHAIN_TEMPLATES[6]  # SUB_MUL
        result = _compute_chain(tmpl, [8, 3, 2])
        assert result == (8 - 3) * 2  # 10

    def test_all_templates_have_unique_ids(self):
        ids = [t["id"] for t in CHAIN_TEMPLATES]
        assert len(ids) == len(set(ids))

    def test_all_templates_have_required_keys(self):
        for tmpl in CHAIN_TEMPLATES:
            assert "id" in tmpl
            assert "name" in tmpl
            assert "ops" in tmpl
            assert "n_args" in tmpl
            assert "compute" in tmpl


class TestMultiStepBatcher:
    def test_init(self):
        batcher = MultiStepBatcher()
        assert batcher.max_num == 10
        assert batcher.output_dim == 128

    def test_sample_one(self):
        batcher = MultiStepBatcher()
        tokens, task_id, label = batcher.sample_one()
        assert isinstance(tokens, list)
        assert len(tokens) == batcher.max_len
        assert tokens[0] == CLS
        assert task_id >= CHAIN_ID_OFFSET
        assert 0 <= label < batcher.output_dim

    def test_sample_one_pad(self):
        batcher = MultiStepBatcher()
        tokens, _, _ = batcher.sample_one()
        # Should have at least one PAD at end
        assert tokens[-1] == PAD or tokens[-2] == PAD or tokens[-3] == PAD

    def test_batch(self):
        batcher = MultiStepBatcher()
        batch = batcher.batch(16)
        assert batch["input_ids"].shape == (16, batcher.max_len)
        assert batch["task_ids"].shape == (16,)
        assert batch["labels"].shape == (16,)
        assert batch["input_ids"].dtype == torch.long
        assert batch["task_ids"].dtype == torch.long

    def test_batch_device(self):
        batcher = MultiStepBatcher()
        batch = batcher.batch(8, device="cpu")
        assert batch["input_ids"].device.type == "cpu"

    def test_num_tasks(self):
        batcher = MultiStepBatcher()
        assert batcher.num_tasks == CHAIN_ID_OFFSET + len(CHAIN_TEMPLATES)

    def test_task_id_range(self):
        batcher = MultiStepBatcher()
        batch = batcher.batch(64)
        task_ids = batch["task_ids"].unique()
        for tid in task_ids:
            assert tid.item() >= CHAIN_ID_OFFSET
            assert tid.item() < CHAIN_ID_OFFSET + len(CHAIN_TEMPLATES)

    def test_label_range(self):
        batcher = MultiStepBatcher()
        batch = batcher.batch(64)
        assert batch["labels"].min() >= 0
        assert batch["labels"].max() < 128
