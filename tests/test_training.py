import pytest
import tempfile
from pathlib import Path
from hts.config import HtSConfig, TransformerConfig, TrainConfig
from hts.training import train_synthetic, evaluate, build_model, set_seed
from hts.data.synthetic_tasks import SyntheticTaskBatcher, FAMILIES


class TestSetSeed:
    def test_reproducibility(self):
        import torch
        set_seed(42)
        a = torch.randn(4)
        set_seed(42)
        b = torch.randn(4)
        assert (a == b).all()


class TestBuildModel:
    def test_build_hts(self):
        model = build_model("hts")
        assert hasattr(model, "blocks")

    def test_build_transformer(self):
        model = build_model("transformer")
        assert hasattr(model, "blocks")

    def test_build_with_configs(self):
        hts_cfg = HtSConfig(d_model=24, dim_ff=32)
        tf_cfg = TransformerConfig(d_model=24, dim_ff=32)
        model_hts = build_model("hts", hts_config=hts_cfg)
        model_tf = build_model("transformer", tf_config=tf_cfg)
        assert model_hts is not None
        assert model_tf is not None

    def test_invalid_model_kind(self):
        with pytest.raises(ValueError, match="Unknown model kind"):
            build_model("invalid")


class TestEvaluate:
    def test_evaluate_hts(self):
        model = build_model("hts")
        batcher = SyntheticTaskBatcher()
        ev = evaluate(model, batcher, "cpu", batch_size=8, eval_batches=2)
        assert "macro_acc" in ev
        assert "macro_loss" in ev
        for family in FAMILIES:
            assert f"{family}_acc" in ev

    def test_evaluate_transformer(self):
        model = build_model("transformer")
        batcher = SyntheticTaskBatcher()
        ev = evaluate(model, batcher, "cpu", batch_size=8, eval_batches=2)
        assert "macro_acc" in ev
        assert "macro_loss" in ev


class TestTrainSynthetic:
    def test_train_hts_short(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_synthetic(
                model_kind="hts",
                train_config=TrainConfig(steps=5, batch_size=8, eval_every=5, eval_batches=2),
                hts_config=HtSConfig(d_model=24, dim_ff=32, rank_main=3, rank_corr=1),
                out_dir=tmpdir,
            )
            assert "history" in result
            assert "meta" in result
            assert "metrics_path" in result
            assert len(result["history"]) > 0
            assert Path(result["metrics_path"]).exists()

    def test_train_transformer_short(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_synthetic(
                model_kind="transformer",
                train_config=TrainConfig(steps=5, batch_size=8, eval_every=5, eval_batches=2),
                tf_config=TransformerConfig(d_model=24, dim_ff=32),
                out_dir=tmpdir,
            )
            assert "history" in result
            assert len(result["history"]) > 0

    def test_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_synthetic(
                model_kind="hts",
                train_config=TrainConfig(steps=3, batch_size=8, eval_every=1, eval_batches=1),
                hts_config=HtSConfig(d_model=24, dim_ff=32, rank_main=3, rank_corr=1),
                out_dir=tmpdir,
            )
            out = Path(tmpdir)
            assert (out / "hts_state_dict.pt").exists()
            assert (out / "hts_meta.json").exists()
            assert (out / "hts_metrics.csv").exists()

    def test_meta_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_synthetic(
                model_kind="hts",
                train_config=TrainConfig(steps=3, batch_size=8, eval_every=1, eval_batches=1),
                hts_config=HtSConfig(d_model=24, dim_ff=32, rank_main=3, rank_corr=1),
                out_dir=tmpdir,
            )
            meta = result["meta"]
            assert meta["model_kind"] == "hts"
            assert "params" in meta
            assert "device" in meta
            assert "train_config" in meta


class TestCLIMain:
    def test_cli_basic(self):
        from hts.cli import main
        with tempfile.TemporaryDirectory() as tmpdir:
            main(["--model", "hts", "--device", "cpu", "--steps", "3",
                   "--batch-size", "8", "--eval-every", "1", "--eval-batches", "1",
                   "--out-dir", tmpdir])
