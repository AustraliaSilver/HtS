"""Train HtS and Transformer baseline with the same public interface."""
from hts import HtSConfig, TransformerConfig, TrainConfig
from hts.training import train_synthetic

cfg = TrainConfig(steps=120, batch_size=64, eval_every=40, eval_batches=5, device="auto", seed=42)
train_synthetic("transformer", train_config=cfg, tf_config=TransformerConfig(), out_dir="runs/compare_transformer")
train_synthetic("hts", train_config=cfg, hts_config=HtSConfig(), out_dir="runs/compare_hts")
