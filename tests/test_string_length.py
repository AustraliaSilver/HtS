"""
Tests for string_length benchmark.
"""
import pytest
import torch
from hts.data.string_length_tasks import (
    StringLengthBatcher, TASKS, FAMILIES, CHAR_TOKENS,
    _generate_string, _string_to_tokens, _label_for_task, _tok,
    PAD, CLS, SEP, NUM_OFFSET, VOCAB_SIZE, OUTPUT_DIM, MAX_LEN, LENGTH_OFFSET,
    build_string_length_batcher
)


class TestConstants:
    def test_pad_value(self):
        assert PAD == 0

    def test_cls_value(self):
        assert CLS == 1

    def test_sep_value(self):
        assert SEP == 2

    def test_num_offset(self):
        assert NUM_OFFSET == 10

    def test_vocab_size(self):
        assert VOCAB_SIZE == 64

    def test_output_dim(self):
        assert OUTPUT_DIM == 128

    def test_max_len(self):
        assert MAX_LEN == 200

    def test_tasks_count(self):
        assert len(TASKS) == 10

    def test_families_count(self):
        assert len(FAMILIES) == 3

    def test_char_tokens_count(self):
        assert len(CHAR_TOKENS) >= 36  # At least 26 letters + 10 digits


class TestGenerateString:
    def test_lowercase_string(self):
        s = _generate_string(0, min_len=3, max_len=5)
        assert len(s) >= 3 and len(s) <= 5
        assert all(c in 'abcdefghijklmnopqrstuvwxyz' for c in s)

    def test_digit_string(self):
        s = _generate_string(1, min_len=3, max_len=5)
        assert len(s) >= 3 and len(s) <= 5
        assert all(c in '0123456789' for c in s)

    def test_mixed_string(self):
        s = _generate_string(2, min_len=3, max_len=5)
        assert len(s) >= 3 and len(s) <= 5
        assert all(c.isalnum() for c in s)

    def test_string_with_spaces(self):
        s = _generate_string(3, min_len=3, max_len=10)
        assert len(s) >= 3
        assert ' ' in s


class TestStringToTokens:
    def test_basic_conversion(self):
        tokens = _string_to_tokens("abc", 5)
        assert len(tokens) == 5
        assert tokens[0] == CHAR_TOKENS['a']
        assert tokens[1] == CHAR_TOKENS['b']
        assert tokens[2] == CHAR_TOKENS['c']
        assert tokens[3] == PAD
        assert tokens[4] == PAD

    def test_truncation(self):
        tokens = _string_to_tokens("abcdef", 3)
        assert len(tokens) == 3
        assert tokens[0] == CHAR_TOKENS['a']
        assert tokens[1] == CHAR_TOKENS['b']
        assert tokens[2] == CHAR_TOKENS['c']

    def test_padding(self):
        tokens = _string_to_tokens("ab", 6)
        assert len(tokens) == 6
        assert tokens[0] == CHAR_TOKENS['a']
        assert tokens[1] == CHAR_TOKENS['b']
        assert tokens[2:] == [PAD] * 4


class TestLabelForTask:
    def test_len_lowercase(self):
        s = "hello"
        label = _label_for_task(0, s)
        assert label == 5

    def test_len_digits(self):
        s = "12345"
        label = _label_for_task(1, s)
        assert label == 5

    def test_len_mixed(self):
        s = "abc123"
        label = _label_for_task(2, s)
        assert label == 6

    def test_count_vowels(self):
        s = "hello"
        label = _label_for_task(4, s)
        assert label == 2  # e, o

    def test_count_consonants(self):
        s = "hello"
        label = _label_for_task(5, s)
        assert label == 3  # h, l, l

    def test_count_digits(self):
        s = "abc123"
        label = _label_for_task(6, s)
        assert label == 3

    def test_has_vowel(self):
        assert _label_for_task(7, "hello") == 1
        assert _label_for_task(7, "bcdfg") == 0

    def test_first_char_type(self):
        assert _label_for_task(8, "abc") == 0  # letter
        assert _label_for_task(8, "123") == 1  # digit

    def test_last_char_type(self):
        assert _label_for_task(9, "abc") == 0  # letter
        assert _label_for_task(9, "123") == 1  # digit


class TestStringLengthBatcher:
    def test_init(self):
        batcher = StringLengthBatcher(max_len=20, output_dim=128, num_tasks=10)
        assert batcher.max_len == 20
        assert batcher.output_dim == 128
        assert batcher.num_tasks == 10

    def test_sample_batch_shapes(self):
        batcher = StringLengthBatcher(max_len=20, num_tasks=10)
        x, y, task_ids = batcher.sample_batch(32)
        assert x.shape == (32, 20)
        assert y.shape == (32,)
        assert task_ids.shape == (32,)

    def test_sample_batch_dtypes(self):
        batcher = StringLengthBatcher(max_len=20, num_tasks=10)
        x, y, task_ids = batcher.sample_batch(16)
        assert x.dtype == torch.long
        assert y.dtype == torch.long
        assert task_ids.dtype == torch.long

    def test_sample_batch_value_ranges(self):
        batcher = StringLengthBatcher(max_len=20, num_tasks=10)
        x, y, task_ids = batcher.sample_batch(64)
        assert x.min() >= 0
        assert x.max() < VOCAB_SIZE
        assert y.min() >= 0
        assert y.max() < OUTPUT_DIM
        assert task_ids.min() >= 0
        assert task_ids.max() < 10

    def test_sample_batch_with_meta(self):
        batcher = StringLengthBatcher(max_len=20, num_tasks=10)
        meta = batcher.sample_batch_with_meta(16)
        assert 'x' in meta
        assert 'y' in meta
        assert 'task_ids' in meta
        assert 'task_names' in meta
        assert len(meta['task_names']) == 16

    def test_build_string_length_batcher(self):
        batcher = build_string_length_batcher(num_tasks=10, seed=42)
        assert isinstance(batcher, StringLengthBatcher)
        assert batcher.num_tasks == 10

    def test_reproducibility(self):
        # Test that same seed produces same task sequence
        batcher1 = StringLengthBatcher(max_len=20, num_tasks=10, seed=42)
        batcher2 = StringLengthBatcher(max_len=20, num_tasks=10, seed=42)
        # Both should have same internal state
        assert batcher1.rng.randint(0, 100) == batcher2.rng.randint(0, 100)

    def test_different_seeds(self):
        batcher1 = StringLengthBatcher(max_len=20, num_tasks=10, seed=42)
        batcher2 = StringLengthBatcher(max_len=20, num_tasks=10, seed=123)
        x1, _, _ = batcher1.sample_batch(32)
        x2, _, _ = batcher2.sample_batch(32)
        assert not torch.equal(x1, x2)


class TestBatchIntegration:
    def test_batch_contains_valid_tokens(self):
        batcher = StringLengthBatcher(max_len=20, num_tasks=10)
        x, y, task_ids = batcher.sample_batch(32)
        # All tokens should be valid
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                assert x[i, j].item() in range(VOCAB_SIZE)

    def test_batch_labels_match_tasks(self):
        batcher = StringLengthBatcher(max_len=20, num_tasks=10)
        x, y, task_ids = batcher.sample_batch(32)
        # Labels should be valid for the output dimension
        assert all(0 <= label.item() < OUTPUT_DIM for label in y)

    def test_batch_task_ids_valid(self):
        batcher = StringLengthBatcher(max_len=20, num_tasks=10)
        x, y, task_ids = batcher.sample_batch(32)
        # All task IDs should be valid
        assert all(0 <= tid.item() < 10 for tid in task_ids)