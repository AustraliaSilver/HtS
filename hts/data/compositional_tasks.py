"""Compositional task arithmetic benchmark for HtS architecture.

Each task is a composition of two primitive operations:
  COMPOSE(OP1, OP2)(a, b, c) = OP2(OP1(a, b), c)

This tests:
  - Task compositionality (can model compose known primitives?)
  - Transfer learning (does knowing OP1 help learn COMPOSE(OP1, OP2)?)
  - Generalization to unseen compositions

Base primitives: add, sub, mul, max, min, mod, gt, parity
Compositions: all 8×7 = 56 possible pairs (OP1 ≠ OP2)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import torch

PAD = 0
CLS = 1
SEP = 2
NUM_OFFSET = 10
TOKEN_THEN = 5  # separator between OP1 and OP2

# Compositional task IDs start at 200 to avoid collision
COMP_ID_OFFSET = 200

# Base primitives
BASE_OPS = {
    "add":    lambda a, b: a + b,
    "sub":    lambda a, b: max(0, a - b),
    "mul":    lambda a, b: a * b,
    "max":    lambda a, b: max(a, b),
    "min":    lambda a, b: min(a, b),
    "mod":    lambda a, b: a % max(1, b),
    "gt":     lambda a, b: int(a > b),
    "parity": lambda a, b: (a + b) % 2,
}

OP_NAMES = list(BASE_OPS.keys())

# Generate all valid compositions (OP1 ≠ OP2)
COMPOSITIONS = []
_comp_id = 0
for op1 in OP_NAMES:
    for op2 in OP_NAMES:
        if op1 != op2:
            COMPOSITIONS.append({
                "id": _comp_id,
                "op1": op1,
                "op2": op2,
                "fn1": BASE_OPS[op1],
                "fn2": BASE_OPS[op2],
            })
            _comp_id += 1


def _tok(n: int) -> int:
    return NUM_OFFSET + int(max(0, min(50, n)))


@dataclass
class CompositionalBatcher:
    """Generate compositional task arithmetic batches.

    Each sample: COMPOSE(OP1, OP2)(a, b, c) = OP2(OP1(a, b), c)
    Input tokens: [CLS] a b c THEN OP1 OP2 [SEP]
    """
    max_num: int = 10
    output_dim: int = 128
    vocab_size: int = 64
    max_len: int = 12

    def sample_one(self) -> Tuple[List[int], int, int]:
        comp = random.choice(COMPOSITIONS)
        a = random.randint(0, self.max_num)
        b = random.randint(0, self.max_num)
        c = random.randint(0, self.max_num)

        intermediate = comp["fn1"](a, b)
        result = comp["fn2"](intermediate, c)

        # Tokens: [CLS] a b c THEN OP1 OP2 [SEP]
        op1_idx = OP_NAMES.index(comp["op1"])
        op2_idx = OP_NAMES.index(comp["op2"])
        tokens = [CLS, _tok(a), _tok(b), _tok(c), TOKEN_THEN, NUM_OFFSET + op1_idx, NUM_OFFSET + op2_idx, SEP]
        tokens = tokens[: self.max_len]
        tokens += [PAD] * (self.max_len - len(tokens))

        task_id = COMP_ID_OFFSET + comp["id"]
        label = int(result % self.output_dim)
        return tokens, task_id, label

    def batch(self, batch_size: int, device=None) -> Dict[str, torch.Tensor]:
        rows = [self.sample_one() for _ in range(batch_size)]
        return {
            "input_ids": torch.tensor([r[0] for r in rows], dtype=torch.long, device=device),
            "task_ids": torch.tensor([r[1] for r in rows], dtype=torch.long, device=device),
            "labels": torch.tensor([r[2] for r in rows], dtype=torch.long, device=device),
        }

    @property
    def num_tasks(self) -> int:
        return COMP_ID_OFFSET + len(COMPOSITIONS)
