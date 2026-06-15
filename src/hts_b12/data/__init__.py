from .string_tasks import StringBatch, TASKS, TOKENS, make_string_count_batch
from .prime_tasks import PrimeBatch, PrimeDataset, is_prime, make_prime_batch, encode_number

__all__ = [
    "StringBatch", "TASKS", "TOKENS", "make_string_count_batch",
    "PrimeBatch", "PrimeDataset", "is_prime", "make_prime_batch", "encode_number",
]
