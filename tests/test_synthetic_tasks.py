import torch
import pytest
from hts.data.synthetic_tasks import (
    SyntheticTaskBatcher,
    FAMILIES,
    TASKS,
    VOCAB_SIZE,
    OUTPUT_DIM,
    MAX_LEN,
    _label_for_task,
)


class TestSyntheticTaskBatcher:
    def test_sample_one(self):
        batcher = SyntheticTaskBatcher()
        tokens, task, label = batcher.sample_one("arith8")
        assert len(tokens) == MAX_LEN
        assert 0 <= task < 24
        assert 0 <= label < OUTPUT_DIM

    def test_batch_shape(self):
        batcher = SyntheticTaskBatcher()
        batch = batcher.batch(16, family="arith8")
        assert batch["input_ids"].shape == (16, MAX_LEN)
        assert batch["task_ids"].shape == (16,)
        assert batch["labels"].shape == (16,)

    def test_batch_device(self):
        batcher = SyntheticTaskBatcher()
        batch = batcher.batch(8, family="arith8", device="cpu")
        assert str(batch["input_ids"].device) == "cpu"

    def test_all_families(self):
        batcher = SyntheticTaskBatcher()
        for family in FAMILIES:
            batch = batcher.batch(4, family=family)
            assert batch["input_ids"].shape == (4, MAX_LEN)

    def test_random_family(self):
        batcher = SyntheticTaskBatcher()
        batch = batcher.batch(16, family=None)
        assert batch["input_ids"].shape == (16, MAX_LEN)

    def test_invalid_family(self):
        batcher = SyntheticTaskBatcher()
        with pytest.raises(ValueError, match="Unknown family"):
            batcher.sample_one("invalid")

    def test_ood_range(self):
        batcher = SyntheticTaskBatcher()
        for _ in range(20):
            tokens, task, label = batcher.sample_one("arith_ood")
            nums = [t for t in tokens if t >= 10]
            for n in nums:
                assert n >= batcher.min_num_ood + 10


class TestLabelForTask:
    def test_add(self):
        assert _label_for_task(0, [3, 5]) == 8

    def test_mul(self):
        assert _label_for_task(1, [3, 5]) == 15

    def test_max(self):
        assert _label_for_task(2, [3, 5]) == 5

    def test_min(self):
        assert _label_for_task(3, [3, 5]) == 3

    def test_abs(self):
        assert _label_for_task(4, [3, 5]) == 2

    def test_mod(self):
        assert _label_for_task(5, [7, 3]) == 1

    def test_gt(self):
        assert _label_for_task(6, [3, 5]) == 0
        assert _label_for_task(6, [5, 3]) == 1

    def test_parity(self):
        assert _label_for_task(7, [4, 6]) == 0
        assert _label_for_task(7, [4, 5]) == 1

    def test_first(self):
        assert _label_for_task(8, [3, 5]) == 3

    def test_last(self):
        assert _label_for_task(9, [3, 5]) == 5

    def test_summod10(self):
        assert _label_for_task(10, [3, 5, 7]) == 5

    def test_count_even(self):
        assert _label_for_task(11, [2, 3, 4, 5]) == 2

    def test_seq_max(self):
        assert _label_for_task(12, [3, 5, 7, 1]) == 7

    def test_seq_min(self):
        assert _label_for_task(13, [3, 5, 7, 1]) == 1

    def test_is_pal(self):
        assert _label_for_task(14, [1, 2, 1]) == 1
        assert _label_for_task(14, [1, 2, 3]) == 0

    def test_gt3_count(self):
        assert _label_for_task(15, [1, 2, 4, 5]) == 2

    def test_sum3(self):
        assert _label_for_task(16, [1, 2, 3]) == 6

    def test_muladd(self):
        assert _label_for_task(17, [2, 3, 4]) == 10

    def test_ifgt(self):
        assert _label_for_task(18, [5, 3, 1]) == 5
        assert _label_for_task(18, [1, 3, 5]) == 5

    def test_modmul(self):
        # ((a+1)*(b+1)) % max(1, c+1) = ((1+1)*(2+1)) % max(1, 3+1) = 6 % 4 = 2
        assert _label_for_task(19, [1, 2, 3]) == 2

    def test_parity3(self):
        assert _label_for_task(20, [1, 2, 3]) == 0

    def test_chaincmp(self):
        assert _label_for_task(21, [3, 2, 1]) == 1
        assert _label_for_task(21, [1, 3, 2]) == 0

    def test_max3(self):
        assert _label_for_task(22, [1, 3, 2]) == 3

    def test_min3(self):
        assert _label_for_task(23, [1, 3, 2]) == 1


class TestConstants:
    def test_vocab_size(self):
        assert VOCAB_SIZE == 64

    def test_output_dim(self):
        assert OUTPUT_DIM == 128

    def test_max_len(self):
        assert MAX_LEN == 12

    def test_task_count(self):
        assert len(TASKS) == 24

    def test_family_count(self):
        assert len(FAMILIES) == 4
