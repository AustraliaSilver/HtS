"""Multi-step reasoning chain benchmark for HtS architecture.

Each task is a chain of 2-4 primitive operations where each step depends
on the result of the previous step. This tests:
  - Depth reasoning (3-5 computation steps)
  - Intermediate state tracking
  - Task-specific routing through computation chains

Chain templates:
  - ADD then MUL:          (a + b) * c
  - MAX then SUB:          max(a, b) - c
  - ADD then ADD then MOD: ((a + b) + c) % d
  - IFGT then ADD:         if a > b then c + d else c - d
  - MUL then MOD then ADD: ((a * b) % c) + d
  - ADD then MAX then MIN: min(max(a + b, c), d)
  - SUB then MUL:          (a - b) * c
  - IFGT then MUL:         if a > b then c * d else c + d
  - ADD then ADD then ADD: a + b + c + d
  - MAX then MAX:          max(max(a, b), c)
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
TOKEN_OP = 4   # token markers for operations
TOKEN_THEN = 5
TOKEN_IF = 6
TOKEN_THEN2 = 7
TOKEN_ELSE = 8
TOKEN_RES = 9

MAX_CHAIN_LEN = 10  # max tokens in chain expression
MAX_TOTAL_LEN = 18  # max total input length including CLS/SEP

# Chain task IDs start at 100 to avoid collision with base tasks (0-23)
CHAIN_ID_OFFSET = 100

CHAIN_TEMPLATES = [
    {
        "id": 0,
        "name": "ADD_MUL",
        "ops": ["add", "mul"],
        "n_args": 3,
        "compute": lambda args: (args[0] + args[1]) * args[2],
    },
    {
        "id": 1,
        "name": "MAX_SUB",
        "ops": ["max", "sub"],
        "n_args": 3,
        "compute": lambda args: max(args[0], args[1]) - args[2],
    },
    {
        "id": 2,
        "name": "ADD_ADD_MOD",
        "ops": ["add", "add", "mod"],
        "n_args": 4,
        "compute": lambda args: (args[0] + args[1] + args[2]) % max(1, args[3]),
    },
    {
        "id": 3,
        "name": "IFGT_ADD",
        "ops": ["ifgt", "add"],
        "n_args": 4,
        "compute": lambda args: (args[2] + args[3]) if args[0] > args[1] else (args[2] - args[3]),
    },
    {
        "id": 4,
        "name": "MUL_MOD_ADD",
        "ops": ["mul", "mod", "add"],
        "n_args": 4,
        "compute": lambda args: ((args[0] * args[1]) % max(1, args[2])) + args[3],
    },
    {
        "id": 5,
        "name": "ADD_MAX_MIN",
        "ops": ["add", "max", "min"],
        "n_args": 4,
        "compute": lambda args: min(max(args[0] + args[1], args[2]), args[3]),
    },
    {
        "id": 6,
        "name": "SUB_MUL",
        "ops": ["sub", "mul"],
        "n_args": 3,
        "compute": lambda args: max(0, args[0] - args[1]) * args[2],
    },
    {
        "id": 7,
        "name": "IFGT_MUL",
        "ops": ["ifgt", "mul"],
        "n_args": 4,
        "compute": lambda args: (args[2] * args[3]) if args[0] > args[1] else (args[2] + args[3]),
    },
    {
        "id": 8,
        "name": "ADD_ADD_ADD",
        "ops": ["add", "add", "add"],
        "n_args": 4,
        "compute": lambda args: args[0] + args[1] + args[2] + args[3],
    },
    {
        "id": 9,
        "name": "MAX_MAX",
        "ops": ["max", "max"],
        "n_args": 3,
        "compute": lambda args: max(max(args[0], args[1]), args[2]),
    },
]


def _tok(n: int) -> int:
    return NUM_OFFSET + int(max(0, min(50, n)))


def _compute_chain(template: dict, args: List[int]) -> int:
    return template["compute"](args)


@dataclass
class MultiStepBatcher:
    """Generate multi-step reasoning chain batches.

    Each sample is a chain of 2-4 primitive operations where each step
    depends on the result of the previous step.
    """
    max_num: int = 10
    output_dim: int = 128
    vocab_size: int = 64
    max_len: int = MAX_TOTAL_LEN

    def sample_one(self) -> Tuple[List[int], int, int]:
        template = random.choice(CHAIN_TEMPLATES)
        n_args = template["n_args"]
        args = [random.randint(0, self.max_num) for _ in range(n_args)]
        result = _compute_chain(template, args)

        # Build token sequence: [CLS] a OP b OP c OP d ... [SEP]
        tokens = [CLS]
        for i, a in enumerate(args):
            tokens.append(_tok(a))
            if i < len(args) - 1:
                tokens.append(TOKEN_OP)
        tokens.append(SEP)

        tokens = tokens[: self.max_len]
        tokens += [PAD] * (self.max_len - len(tokens))
        task_id = CHAIN_ID_OFFSET + template["id"]
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
        return CHAIN_ID_OFFSET + len(CHAIN_TEMPLATES)
