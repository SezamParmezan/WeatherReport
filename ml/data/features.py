import numpy as np
import pandas as pd


def add_cyclic_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour, day-of-week and day-of-year as sin/cos pairs
    so the model understands that hour 23 and hour 0 are close.
    """

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df.index.day_of_year / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.day_of_year / 365)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    return df


def add_lag_features(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    """
    Add lag features for the specified columns.
    For example, if lags=[1, 2], add columns "column_lag_1" and "column_lag_2"
    which contain the column value from 1 hour ago and 2 hours ago, respectively.
    """

    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame:
    """
    Add rolling mean and std for the specified columns.

    Parameters
    - df: DataFrame with a DatetimeIndex (or convertible index).
    - columns: list of column names to compute rolling stats for.
    - windows: list of integer windows in hours (e.g. [6, 24, 168]).

    Adds columns named `{col}_roll_mean_{w}h` and `{col}_roll_std_{w}h`.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            return df

    for col in columns:
        if col not in df.columns:
            continue

        for w in windows:
            win_str = f"{w}H"
            mean_name = f"{col}_roll_mean_{w}h"
            std_name = f"{col}_roll_std_{w}h"

            df[mean_name] = df[col].rolling(win_str, min_periods=1).mean()
            df[std_name] = df[col].rolling(win_str, min_periods=1).std()

    return df


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived physical features when source variables exist:
    - `wind_speed` (from u/v components if available)
    - `relative_humidity` (from temperature and dewpoint)
    - `pressure_diff_3h` (difference with 3 hours ago)
    - log1p of precipitation columns

    This function is robust to missing columns and will only add features
    when the required inputs are present.
    """

    #Wind speed lol
    uv_candidates = [("wind_u", "wind_v"), ("u", "v"), ("u10", "v10"), ("uas", "vas"), ("u_component", "v_component")]
    for u_col, v_col in uv_candidates:
        if u_col in df.columns and v_col in df.columns and "wind_speed" not in df.columns:
            df["wind_speed"] = np.sqrt(df[u_col].fillna(0) ** 2 + df[v_col].fillna(0) ** 2)
            break


    temp_candidates = ["t2m", "temperature", "temp", "t"]
    dew_candidates = ["d2m", "dewpoint", "dew_point", "td"]
    temp_col = next((c for c in temp_candidates if c in df.columns), None)
    dew_col = next((c for c in dew_candidates if c in df.columns), None)
    
    if temp_col and dew_col:
        T = df[temp_col].astype(float) #They are in celsius btw
        D = df[dew_col].astype(float)

        #stuff for vapor pressure
        a, b = 17.625, 243.04
        sat_T = 6.112 * np.exp(a * T / (b + T))
        sat_D = 6.112 * np.exp(a * D / (b + D))

        with np.errstate(divide="ignore", invalid="ignore"):
            rh = 100.0 * (sat_D / sat_T)
        df["relative_humidity"] = rh.clip(0, 100)

    #PRESSURE DIFF 3h
    pressure_candidates = ["msl", "pressure", "p", "pressure_msl", "air_pressure"]
    pres_col = next((c for c in pressure_candidates if c in df.columns), None)
    if pres_col:
        df["pressure_diff_3h"] = df[pres_col] - df[pres_col].shift(3)

    #LOG1P precipitation
    precip_candidates = ["tp", "precipitation", "precip", "rain", "rainfall"]
    for pc in precip_candidates:
        if pc in df.columns:
            out_name = f"{pc}_log1p"
            vals = df[pc].fillna(0)
            vals = vals.where(vals >= 0, 0)
            df[out_name] = np.log1p(vals)

    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single entry point to build all features. Calls other helpers in sequence:
    1) add_cyclic_time
    2) add_physical_features
    3) add_rolling_features (for a selected set)
    4) add_lag_features (for important vars)

    This function is intended to be the only function called from `dataset.py`.
    """

    df = add_cyclic_time(df)
    df = add_physical_features(df)

    rolling_targets = [c for c in ["t2m", "temperature", "msl", "pressure", "wind_speed", "tp", "precipitation"] if c in df.columns]
    if rolling_targets:
        df = add_rolling_features(df, rolling_targets, windows=[6, 24, 168])

    lag_targets = [c for c in ["temperature", "wind_speed", "precipitation", "pressure_msl"] if c in df.columns]
    if lag_targets:
        df = add_lag_features(df, lag_targets, lags=[1, 2, 3, 24])

    return df


