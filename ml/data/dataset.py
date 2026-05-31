import pandas as pd

from pathlib import Path
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer



def load_and_process_data(dir_path: Path) -> pd.DataFrame:
    """It loads all processed files in directory and concatenates them into a single DataFrame, 
    sorted by index (datetime)
    """

    files = sorted(dir_path.glob("rome_*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs).sort_index()
    return df


def split_data(df: pd.DataFrame):
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    return train, val, test


def make_timeseries_dataset(df: pd.DataFrame, encoder_len: int, prediction_len: int) -> TimeSeriesDataSet:
    df = df.copy()
    df["location"] = "rome"
    df["time_idx"] = range(len(df))

    '''It will return a TimeSeriesDataSet object that can be used for training a PyTorch Forecasting model'''

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="temperature",
        group_ids=["location"],
        min_encoder_length=encoder_len,
        max_encoder_length=encoder_len,
        min_prediction_length=prediction_len,
        max_prediction_length=prediction_len,
        time_varying_known_reals=["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"],
        time_varying_unknown_reals=[
                "temperature", "pressure_msl", "pressure_surf",
                "wind_u", "wind_v", "wind_speed",
                "dewpoint_temp", "cloud_cover", "precipitation",
                "pressure_diff_3h", "relative_humidity", "precipitation_log1p",
                "temperature_lag_1", "temperature_lag_3", "temperature_lag_6",
                "temperature_lag_12", "temperature_lag_24", "temperature_lag_48", "temperature_lag_168",
                "pressure_msl_lag_1", "pressure_msl_lag_3", "pressure_msl_lag_6",
                "pressure_msl_lag_12", "pressure_msl_lag_24", "pressure_msl_lag_48", "pressure_msl_lag_168",
                "wind_speed_lag_1", "wind_speed_lag_24",
                "precipitation_lag_1", "precipitation_lag_24",
                "temperature_roll_mean_6h", "temperature_roll_std_6h",
                "temperature_roll_mean_24h", "temperature_roll_std_24h",
                "temperature_roll_mean_168h", "temperature_roll_std_168h",
                "pressure_msl_roll_mean_6h", "pressure_msl_roll_mean_24h",
                "precipitation_roll_mean_24h", "precipitation_roll_mean_168h",
            ],
        target_normalizer=GroupNormalizer(groups=["location"]),
        add_relative_time_idx=True,
    )


def make_val_test_datasets(train_dataset: TimeSeriesDataSet, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Create val and test datasets using the same parameters and normalizer
    as the train dataset — prevents data leakage.
    """
    
    val_dataset  = TimeSeriesDataSet.from_dataset(train_dataset, val_df,  predict=True)
    test_dataset = TimeSeriesDataSet.from_dataset(train_dataset, test_df, predict=True)
    return val_dataset, test_dataset