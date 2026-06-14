"""Minimal example: train HtS-B12 on synthetic task-conditioned data."""
from hts import HtSConfig, TrainConfig
from hts.training import train_synthetic

if __name__ == "__main__":
    train_synthetic(
        model_kind="hts",
        hts_config=HtSConfig(),
        train_config=TrainConfig(steps=100, batch_size=64, eval_every=25, device="auto"),
        out_dir="runs/example_hts",
    )
