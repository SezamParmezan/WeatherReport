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


def test_add_cyclic_time_basic():
    mod = load_module(Path("ml/data/features.py"))
    idx = pd.date_range("2020-01-01", periods=24, freq="H")
    df = pd.DataFrame(index=idx)
    out = mod.add_cyclic_time(df.copy())
    # hour 0 -> sin 0, cos 1
    assert np.isclose(out.loc[out.index[0], "hour_sin"], 0)
    assert np.isclose(out.loc[out.index[0], "hour_cos"], 1)


def test_add_lag_features():
    mod = load_module(Path("ml/data/features.py"))
    idx = pd.date_range("2020-01-01", periods=4, freq="H")
    df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=idx)
    out = mod.add_lag_features(df.copy(), ["a"], [1, 2])
    assert "a_lag_1" in out.columns and "a_lag_2" in out.columns
    assert np.isnan(out.iloc[0]["a_lag_1"]) and out.iloc[1]["a_lag_1"] == 1


def test_add_rolling_features_datetime_index():
    mod = load_module(Path("ml/data/features.py"))
    idx = pd.date_range("2020-01-01", periods=4, freq="H")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    out = mod.add_rolling_features(df.copy(), ["a"], [2])
    assert "a_roll_mean_2h" in out.columns
    # expected means: [1, 1.5, 2.5, 3.5]
    expected = [1.0, 1.5, 2.5, 3.5]
    assert np.allclose(out["a_roll_mean_2h"].values, expected, atol=1e-6)


def test_add_rolling_features_nonconvertible_index_returns_same():
    mod = load_module(Path("ml/data/features.py"))
    df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
    out = mod.add_rolling_features(df.copy(), ["a"], [6])
    # no rolling columns created because index conversion fails
    assert not any("roll_mean" in c or "roll_std" in c for c in out.columns)


def test_add_physical_features_basic():
    mod = load_module(Path("ml/data/features.py"))
    idx = pd.date_range("2020-01-01", periods=3, freq="H")
    df = pd.DataFrame({
        "u": [3.0, 0.0, np.nan],
        "v": [4.0, 0.0, np.nan],
        "temperature": [20.0, 15.0, 10.0],
        "dewpoint": [10.0, 5.0, 0.0],
        "msl": [1010.0, 1009.0, 1008.5],
        "tp": [0.1, -1.0, 2.0],
    }, index=idx)

    out = mod.add_physical_features(df.copy())
    assert "wind_speed" in out.columns
    assert np.isclose(out.loc[idx[0], "wind_speed"], 5.0)
    assert "relative_humidity" in out.columns
    assert (out["relative_humidity"].between(0, 100)).all()
    assert "pressure_diff_3h" in out.columns
    assert "tp_log1p" in out.columns
    # negative precipitation clamped to 0 -> log1p(0) == 0
    assert np.isclose(out.loc[idx[1], "tp_log1p"], 0.0)


def test_build_all_features_compose():
    mod = load_module(Path("ml/data/features.py"))
    idx = pd.date_range("2020-01-01", periods=10, freq="H")
    df = pd.DataFrame({
        "temperature": np.linspace(10, 19, 10),
        "dewpoint": np.linspace(5, 14, 10),
        "msl": np.linspace(1000, 1009, 10),
        "tp": [0.0] * 10,
        "u": [1.0] * 10,
        "v": [1.0] * 10,
    }, index=idx)

    out = mod.build_all_features(df.copy())
    # expect cyclic cols, physical cols, rolling and lag cols
    assert "hour_sin" in out.columns and "hour_cos" in out.columns
    assert "wind_speed" in out.columns and "relative_humidity" in out.columns
    assert any(c.endswith("_roll_mean_6h") for c in out.columns)
    assert any(c.endswith("_lag_1") for c in out.columns)
