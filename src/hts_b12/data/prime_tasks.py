"""Prime number prediction task for HtS-B12.

Generates batches of numbers encoded as digit sequences with binary
prime/not-prime labels. Designed for testing HtS-B12 on a simple but
non-trivial classification task.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


PAD = 0
DIGITS = {str(d): d + 1 for d in range(10)}
NUM_DIGITS = 11  # 0-9 + PAD


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def encode_number(n: int, max_length: int = 6) -> List[int]:
    s = str(n)
    tokens = [DIGITS[c] for c in s]
    if len(tokens) < max_length:
        tokens = [PAD] * (max_length - len(tokens)) + tokens
    return tokens[:max_length]


def sieve_primes(limit: int) -> List[bool]:
    is_p = [False] * (limit + 1)
    if limit >= 2:
        is_p[2] = True
    for i in range(3, limit + 1, 2):
        is_p[i] = True
    for i in range(3, int(math.isqrt(limit)) + 1, 2):
        if is_p[i]:
            for j in range(i * i, limit + 1, i * 2):
                is_p[j] = False
    return is_p


@dataclass
class PrimeBatch:
    input_ids: torch.Tensor
    task_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


class PrimeDataset:
    def __init__(
        self,
        min_val: int = 2,
        max_val: int = 999_999,
        max_length: int = 6,
        seed: int = 42,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.max_length = max_length

        self.prime_flags = sieve_primes(max_val)
        self.numbers = list(range(min_val, max_val + 1))

        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(self.numbers), generator=rng)
        self.numbers = [self.numbers[i] for i in perm.tolist()]

        n = len(self.numbers)
        self.train = self.numbers[: int(n * 0.8)]
        self.val = self.numbers[int(n * 0.8): int(n * 0.9)]
        self.test = self.numbers[int(n * 0.9):]

    def is_prime(self, n: int) -> bool:
        return self.prime_flags[n]

    def make_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device,
        seed: int,
    ) -> PrimeBatch:
        pool = getattr(self, split)
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randint(0, len(pool), (batch_size,), generator=rng)

        ids = torch.zeros(batch_size, self.max_length, dtype=torch.long)
        labels = torch.empty(batch_size, dtype=torch.long)

        for i, idx in enumerate(indices.tolist()):
            n = pool[idx]
            ids[i] = torch.tensor(encode_number(n, self.max_length), dtype=torch.long)
            labels[i] = 1 if self.is_prime(n) else 0

        return PrimeBatch(
            input_ids=ids.to(device),
            task_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            labels=labels.to(device),
        )


def make_prime_batch(
    batch_size: int,
    max_length: int = 6,
    device: torch.device | str = "cpu",
    min_val: int = 2,
    max_val: int = 999_999,
    seed: int = 42,
    split: str = "train",
    _cache: Optional[Dict[int, PrimeDataset]] = None,
) -> PrimeBatch:
    key = max_val
    if _cache is None:
        _cache = {}
    if key not in _cache:
        _cache[key] = PrimeDataset(min_val=min_val, max_val=max_val, max_length=max_length, seed=seed)
    return _cache[key].make_batch(split, batch_size, torch.device(device), seed)
