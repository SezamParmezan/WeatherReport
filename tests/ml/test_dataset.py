import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import types


def load_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_and_process_data_monkeypatched(monkeypatch, tmp_path):
    mod = load_module(Path("ml/data/dataset.py"))

    # Prepare fake parquet files and DataFrames
    idx1 = pd.date_range("2000-01-01", periods=2, freq="H")
    idx2 = pd.date_range("2000-01-01 02:00", periods=2, freq="H")
    df1 = pd.DataFrame({"a": [1, 2]}, index=idx1)
    df2 = pd.DataFrame({"a": [3, 4]}, index=idx2)

    files = [tmp_path / "rome_1.parquet", tmp_path / "rome_2.parquet"]

    def fake_glob(pattern):
        return files

    def fake_read_parquet(path):
        return df1 if path.name.endswith("1.parquet") else df2

    monkeypatch.setattr(Path, "glob", lambda self, p: fake_glob(p))
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    out = mod.load_and_process_data(tmp_path)
    assert isinstance(out, pd.DataFrame)
    assert list(out["a"]) == [1, 2, 3, 4]


def test_split_data():
    import importlib.util, types
    mod = load_module(Path("ml/data/dataset.py"))

    df = pd.DataFrame({"split": ["train", "val", "test"], "x": [1, 2, 3]})
    train, val, test = mod.split_data(df)
    assert len(train) == 1 and train.iloc[0]["split"] == "train"
    assert len(val) == 1 and val.iloc[0]["split"] == "val"
    assert len(test) == 1 and test.iloc[0]["split"] == "test"


def test_make_timeseries_dataset_and_val_test(monkeypatch):
    mod = load_module(Path("ml/data/dataset.py"))

    # Dummy TimeSeriesDataSet replacement
    class DummyTS:
        def __init__(self, df, **kwargs):
            self.df = df
            self.kwargs = kwargs

        @classmethod
        def from_dataset(cls, train_ds, df, predict=False):
            return cls(df, from_train=True, predict=predict)

    monkeypatch.setattr(mod, "TimeSeriesDataSet", DummyTS)

    idx = pd.date_range("2020-01-01", periods=5, freq="H")
    df = pd.DataFrame({"temperature": np.arange(5)}, index=idx)

    ds = mod.make_timeseries_dataset(df, encoder_len=3, prediction_len=2)
    assert isinstance(ds, DummyTS)
    # check that `location` and `time_idx` added
    assert "location" in ds.df.columns
    assert "time_idx" in ds.df.columns

    # make val/test datasets
    val_ds, test_ds = mod.make_val_test_datasets(ds, df, df)
    assert isinstance(val_ds, DummyTS) and isinstance(test_ds, DummyTS)
