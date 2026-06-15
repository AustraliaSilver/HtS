"""Tiny API comparison between HtS-B12 and Transformer baseline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hts_b12 import HtSB12Classifier, HtSB12Config, TransformerClassifier, count_parameters

cfg = HtSB12Config(vocab_size=128, num_tasks=8, num_classes=64, max_length=32, d_model=64, dim_ff=128, num_layers=1)
hts = HtSB12Classifier(cfg)
tf = TransformerClassifier(cfg)
print(f"HtS-B12 params:      {count_parameters(hts):,}")
print(f"Transformer params:  {count_parameters(tf):,}")
