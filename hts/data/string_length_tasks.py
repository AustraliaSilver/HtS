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
MAX_LEN = 200
LENGTH_OFFSET = 30  # Offset for length labels (30-40 = lengths 1-10)

# String character sets
LOWERCASE = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
MIXED = list("abcdefghijklmnopqrstuvwxyz0123456789")

# Task definitions
TASKS = {
    0: "LEN_LOWERCASE",    # Length of lowercase string
    1: "LEN_DIGITS",       # Length of digit string
    2: "LEN_MIXED",        # Length of mixed string
    3: "LEN_WITH_SPACES",  # Length with spaces
    4: "COUNT_VOWELS",     # Count vowels in string
    5: "COUNT_CONSONANTS", # Count consonants
    6: "COUNT_DIGITS",     # Count digits in mixed string
    7: "HAS_VOWEL",        # Check if string has vowel
    8: "FIRST_CHAR_TYPE",  # 0=letter, 1=digit
    9: "LAST_CHAR_TYPE",   # 0=letter, 1=digit
}

FAMILIES = {
    "length": [0, 1, 2, 3],
    "count": [4, 5, 6],
    "binary": [7, 8, 9],
}

# Character to token mapping
CHAR_TOKENS = {
    'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15,
    'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20, 'l': 21,
    'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27,
    's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33,
    'y': 34, 'z': 35,
    '0': 36, '1': 37, '2': 38, '3': 39, '4': 40,
    '5': 41, '6': 42, '7': 43, '8': 44, '9': 45,
    ' ': 46,
}


def _tok(n: int) -> int:
    return NUM_OFFSET + int(max(0, min(50, n)))


def _generate_string(task: int, min_len: int = 2, max_len: int = 100) -> str:
    """Generate a string based on task type."""
    length = random.randint(min_len, max_len)
    
    if task in [0, 4, 5]:  # Lowercase tasks
        s = ''.join(random.choice(LOWERCASE) for _ in range(length))
    elif task in [1, 6]:  # Digit tasks
        s = ''.join(random.choice(DIGITS) for _ in range(length))
    elif task in [2, 7, 8, 9]:  # Mixed tasks
        s = ''.join(random.choice(MIXED) for _ in range(length))
    elif task == 3:  # With spaces
        words = [''.join(random.choice(LOWERCASE) for _ in range(random.randint(1, 10))) 
                 for _ in range(random.randint(2, 10))]
        s = ' '.join(words)
    else:
        s = ''.join(random.choice(LOWERCASE) for _ in range(length))
    
    return s


def _string_to_tokens(s: str, max_len: int) -> List[int]:
    """Convert string to token list."""
    tokens = [CHAR_TOKENS.get(c, 10) for c in s]
    # Pad or truncate to max_len
    if len(tokens) < max_len:
        tokens = tokens + [PAD] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


def _label_for_task(task: int, s: str) -> int:
    """Compute label based on task type."""
    if task == 0:  # LEN_LOWERCASE
        return len(s)
    elif task == 1:  # LEN_DIGITS
        return len(s)
    elif task == 2:  # LEN_MIXED
        return len(s)
    elif task == 3:  # LEN_WITH_SPACES
        return len(s)
    elif task == 4:  # COUNT_VOWELS
        vowels = sum(1 for c in s if c in 'aeiou')
        return min(vowels, OUTPUT_DIM - 1)
    elif task == 5:  # COUNT_CONSONANTS
        consonants = sum(1 for c in s if c.isalpha() and c not in 'aeiou')
        return min(consonants, OUTPUT_DIM - 1)
    elif task == 6:  # COUNT_DIGITS
        digits = sum(1 for c in s if c.isdigit())
        return min(digits, OUTPUT_DIM - 1)
    elif task == 7:  # HAS_VOWEL
        return int(any(c in 'aeiou' for c in s))
    elif task == 8:  # FIRST_CHAR_TYPE
        if len(s) == 0:
            return 0
        return int(s[0].isdigit())
    elif task == 9:  # LAST_CHAR_TYPE
        if len(s) == 0:
            return 0
        return int(s[-1].isdigit())
    else:
        return 0


@dataclass
class StringLengthBatcher:
    """Generate string length prediction benchmark batches.
    
    Tests model's ability to:
    - Predict string length from characters
    - Count specific character types
    - Binary classification (has vowel, char type)
    """
    max_len: int = MAX_LEN
    output_dim: int = OUTPUT_DIM
    num_tasks: int = 10
    seed: int = 42
    
    def __post_init__(self):
        self.rng = random.Random(self.seed)
    
    def sample_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a batch of (x, y, task_ids)."""
        tasks = [self.rng.randint(0, self.num_tasks - 1) for _ in range(n)]
        
        x_list = []
        y_list = []
        
        for task_id in tasks:
            s = _generate_string(task_id, min_len=2, max_len=min(100, self.max_len))
            tokens = _string_to_tokens(s, self.max_len)
            label = _label_for_task(task_id, s)
            
            x_list.append(tokens)
            y_list.append(label)
        
        x = torch.tensor(x_list, dtype=torch.long)
        y = torch.tensor(y_list, dtype=torch.long)
        task_ids = torch.tensor(tasks, dtype=torch.long)
        
        return x, y, task_ids
    
    def batch(self, n: int, device=None) -> Dict:
        """Generate a batch compatible with training loop (returns dict with input_ids, task_ids, labels)."""
        x, y, task_ids = self.sample_batch(n)
        if device is not None:
            x = x.to(device)
            y = y.to(device)
            task_ids = task_ids.to(device)
        return {
            "input_ids": x,
            "task_ids": task_ids,
            "labels": y,
        }

    def sample_batch_with_meta(self, n: int) -> Dict:
        """Generate batch with metadata for diagnostics."""
        x, y, task_ids = self.sample_batch(n)
        
        # Compute per-task accuracy info
        task_names = [TASKS.get(t.item(), f"TASK_{t.item()}") for t in task_ids]
        
        return {
            'x': x,
            'y': y,
            'task_ids': task_ids,
            'task_names': task_names,
        }


def build_string_length_batcher(num_tasks: int = 10, seed: int = 42) -> StringLengthBatcher:
    """Build a string length batcher."""
    return StringLengthBatcher(num_tasks=num_tasks, seed=seed)


# Export for training.py
__all__ = ['StringLengthBatcher', 'TASKS', 'FAMILIES', 'build_string_length_batcher']