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


def test_fit_stats_basic():
    mod = load_module(Path("data/process.py"))
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    stats = mod.fit_stats(df)
    assert isinstance(stats, dict)
    assert np.isclose(stats["a"]["mean"], 2.0)
    assert np.isclose(stats["b"]["std"], np.std([2.0, 4.0, 6.0], ddof=1))


def test_apply_normalization_std0_and_missing():
    mod = load_module(Path("data/process.py"))
    df = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 4.0]})
    stats = {
        "a": {"mean": 1.0, "std": 0.0},
        "b": {"mean": 3.0, "std": 1.0},
    }
    out = mod.apply_normalization(df.copy(), stats)
    # a has std 0 -> set to 0.0
    assert (out["a"] == 0.0).all()
    # b normalized: (2-3)/1 == -1, (4-3)/1 == 1
    assert np.allclose(out["b"].values, [-1.0, 1.0])


def test_clean_interpolate_and_fill():
    mod = load_module(Path("data/process.py"))
    idx = pd.date_range("2020-01-01", periods=5, freq="H")
    df = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 3.0, np.nan], "b": [1.0, np.nan, np.nan, 4.0, 5.0]}, index=idx)
    out = mod.clean(df.copy(), year=2020)
    # no NaNs remain
    assert not out.isna().any().any()
    # values should be numeric
    assert out["a"].dtype.kind in "fi"
    assert out["b"].dtype.kind in "fi"
