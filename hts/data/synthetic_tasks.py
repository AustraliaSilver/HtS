from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import torch

PAD = 0
CLS = 1
SEP = 2
NUM_OFFSET = 10
VOCAB_SIZE = 64
OUTPUT_DIM = 128
MAX_LEN = 12

TASKS = {
    0: "ADD", 1: "MUL", 2: "MAX", 3: "MIN", 4: "ABS", 5: "MOD", 6: "GT", 7: "PARITY",
    8: "FIRST", 9: "LAST", 10: "SUMMOD10", 11: "COUNT_EVEN", 12: "SEQ_MAX", 13: "SEQ_MIN", 14: "IS_PAL", 15: "GT3_COUNT",
    16: "SUM3", 17: "MULADD", 18: "IFGT", 19: "MODMUL", 20: "PARITY3", 21: "CHAINCMP", 22: "MAX3", 23: "MIN3",
}

FAMILIES = {
    "arith8": list(range(0, 8)),
    "seq6": list(range(8, 16)),
    "comp3": list(range(16, 24)),
    "arith_ood": list(range(0, 8)),
}


def _tok(n: int) -> int:
    return NUM_OFFSET + int(max(0, min(50, n)))


def _label_for_task(task: int, nums: List[int]) -> int:
    if task == 0:
        return (nums[0] + nums[1]) % OUTPUT_DIM
    if task == 1:
        return (nums[0] * nums[1]) % OUTPUT_DIM
    if task == 2:
        return max(nums[0], nums[1])
    if task == 3:
        return min(nums[0], nums[1])
    if task == 4:
        return abs(nums[0] - nums[1])
    if task == 5:
        return nums[0] % max(1, nums[1])
    if task == 6:
        return int(nums[0] > nums[1])
    if task == 7:
        return (nums[0] + nums[1]) % 2
    if task == 8:
        return nums[0]
    if task == 9:
        return nums[-1]
    if task == 10:
        return sum(nums) % 10
    if task == 11:
        return sum(1 for x in nums if x % 2 == 0)
    if task == 12:
        return max(nums)
    if task == 13:
        return min(nums)
    if task == 14:
        return int(nums == list(reversed(nums)))
    if task == 15:
        return sum(1 for x in nums if x > 3)
    a, b, c = nums[0], nums[1], nums[2]
    if task == 16:
        return (a + b + c) % OUTPUT_DIM
    if task == 17:
        return (a * b + c) % OUTPUT_DIM
    if task == 18:
        return a if a > b else c
    if task == 19:
        return ((a + 1) * (b + 1)) % max(1, c + 1)
    if task == 20:
        return (a + b + c) % 2
    if task == 21:
        return int((a > b) and (b > c))
    if task == 22:
        return max(a, b, c)
    if task == 23:
        return min(a, b, c)
    raise ValueError(f"Unknown task id: {task}")


@dataclass
class SyntheticTaskBatcher:
    """Generate task-conditioned algorithmic benchmark batches.

    Families:
      - arith8: two-number arithmetic/routing tasks.
      - seq6: six-number sequence tasks.
      - comp3: three-number compositional tasks.
      - arith_ood: two-number tasks with larger numbers than training range.
    """
    max_num_train: int = 10
    max_num_ood: int = 15
    min_num_ood: int = 11
    max_len: int = MAX_LEN
    output_dim: int = OUTPUT_DIM
    vocab_size: int = VOCAB_SIZE

    def sample_one(self, family: str) -> Tuple[List[int], int, int]:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family}. Available: {list(FAMILIES)}")
        task = random.choice(FAMILIES[family])
        if family == "arith_ood":
            nums = [random.randint(self.min_num_ood, self.max_num_ood) for _ in range(6)]
        else:
            nums = [random.randint(0, self.max_num_train) for _ in range(6)]
        if family in {"arith8", "arith_ood"}:
            used = nums[:2]
        elif family == "comp3":
            used = nums[:3]
        else:
            used = nums[:6]
        label = _label_for_task(task, used)
        tokens = [CLS] + [_tok(n) for n in used] + [SEP]
        tokens = tokens[: self.max_len]
        tokens += [PAD] * (self.max_len - len(tokens))
        return tokens, task, int(label % self.output_dim)

    def batch(self, batch_size: int, family: str | None = None, device=None) -> Dict[str, torch.Tensor]:
        families = list(FAMILIES) if family is None else [family]
        rows = [self.sample_one(random.choice(families)) for _ in range(batch_size)]
        return {
            "input_ids": torch.tensor([r[0] for r in rows], dtype=torch.long, device=device),
            "task_ids": torch.tensor([r[1] for r in rows], dtype=torch.long, device=device),
            "labels": torch.tensor([r[2] for r in rows], dtype=torch.long, device=device),
        }
