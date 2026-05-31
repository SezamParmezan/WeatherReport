import pandas as pd
import importlib.util
from pathlib import Path


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DummyDataset:
    def __init__(self, name, df=None):
        self.name = name
        self.df = df
        self.to_dataloader_called = False

    def to_dataloader(self, batch_size, shuffle, num_workers):
        self.to_dataloader_called = True
        assert batch_size == 64
        assert shuffle is False
        assert num_workers == 0
        return f"{self.name}-dataloader"


def test_prepare_data_monkeypatched(monkeypatch):
    mod = load_module(Path("ml/training/train.py"))

    df = pd.DataFrame({"split": ["train", "val", "test"], "temperature": [1.0, 2.0, 3.0]})
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    monkeypatch.setattr(mod, "load_and_process_data", lambda path: df)
    monkeypatch.setattr(mod, "build_all_features", lambda data: data)
    monkeypatch.setattr(mod, "split_data", lambda data: (train_df, val_df, test_df))

    def fake_make_timeseries_dataset(data, encoder_len, prediction_len):
        assert encoder_len == 72
        assert prediction_len == 24
        return DummyDataset("train", data)

    val_ds = DummyDataset("val", val_df)
    test_ds = DummyDataset("test", test_df)

    def fake_make_val_test_datasets(train_dataset, val_df_arg, test_df_arg):
        assert train_dataset.name == "train"
        assert val_df_arg.equals(val_df)
        assert test_df_arg.equals(test_df)
        return val_ds, test_ds

    monkeypatch.setattr(mod, "make_timeseries_dataset", fake_make_timeseries_dataset)
    monkeypatch.setattr(mod, "make_val_test_datasets", fake_make_val_test_datasets)

    train_dataset, train_loader, val_loader = mod.prepare_data()

    assert isinstance(train_dataset, DummyDataset)
    assert train_dataset.name == "train"
    assert train_loader == "train-dataloader"
    assert val_loader == "val-dataloader"
    assert train_dataset.to_dataloader_called is True


def test_build_model_uses_from_dataset(monkeypatch):
    mod = load_module(Path("ml/training/train.py"))

    captured = {}

    def fake_from_dataset(dataset, **kwargs):
        captured["dataset"] = dataset
        captured.update(kwargs)
        return "fake-model"

    monkeypatch.setattr(mod.TemporalFusionTransformer, "from_dataset", classmethod(lambda cls, dataset, **kwargs: fake_from_dataset(dataset, **kwargs)))

    model = mod.build_model("train_dataset")

    assert model == "fake-model"
    assert captured["dataset"] == "train_dataset"
    assert captured["learning_rate"] == 3e-3
    assert captured["hidden_size"] == 64
    assert captured["attention_head_size"] == 4
    assert captured["dropout"] == 0.1
    assert isinstance(captured["loss"], mod.QuantileLoss)
    assert captured["log_interval"] == 10


def test_build_trainer_returns_trainer_and_checkpoint():
    mod = load_module(Path("ml/training/train.py"))

    trainer, checkpoint = mod.build_trainer()

    assert hasattr(trainer, "fit")
    assert trainer.max_epochs == 50
    assert trainer.gradient_clip_val == 0.1
    assert any(type(callback) is type(checkpoint) for callback in trainer.callbacks)
    assert checkpoint.monitor == "val_loss"
    assert checkpoint.save_top_k == 1
    assert checkpoint.mode == "min"
