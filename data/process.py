import os
import argparse
import logging
import json

import xarray as xr
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm


# CONFIG
RAW_DATA_DIR = Path("data/raw/era5")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
STATS_FILE = PROCESSED_DATA_DIR / "stats.json"

YEAR_START = 1996
YEAR_END = 2026

TRAIN_END = 2020
VAL_END = 2022

VARIABLES = {
    "u10": "wind_u",
    "v10": "wind_v",
    "d2m": "dewpoint_temp",
    "t2m": "temperature",
    "msl": "pressure_msl",
    "sp": "pressure_surf",
    "tp": "precipitation",
    "tcc": "cloud_cover",
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_month(nc_path: Path) -> pd.DataFrame:
    """
    Open one monthly .nc file and return a tidy DataFrame.
    Index = DatetimeIndex (hourly), columns = renamed variables.
    ERA5 bbox may contain several lat/lon grid points — averaged spatially.
    """
    ds = xr.open_dataset(nc_path)
    frames = {}

    for nc_var, col_name in VARIABLES.items():
        if nc_var not in ds:
            logger.warning("Variable '%s' not found in %s — filling with NaN", nc_var, nc_path.name)
            frames[col_name] = None
            continue

        da = ds[nc_var]
        spatial_dims = [d for d in da.dims if d in ("latitude", "longitude", "lat", "lon")]
        if spatial_dims:
            da = da.mean(dim=spatial_dims)

        frames[col_name] = da.to_series()

    ds.close()
    df = pd.DataFrame(frames)
    df.index.name = "time"
    return df


def load_year(year: int) -> pd.DataFrame | None:
    """
    Load all 12 monthly files for one year and concatenate into a single DataFrame.
    """
    dfs = []
    for month in range(1, 13):
        nc_path = RAW_DATA_DIR / f"rome_{year}_{month:02d}.nc"
        if not nc_path.exists():
            logger.warning("File %s does not exist, skipping", nc_path)
            continue
        dfs.append(load_month(nc_path))

    if not dfs:
        return None

    df_year = pd.concat(dfs).sort_index()

    n_dupes = df_year.index.duplicated().sum()
    if n_dupes:
        logger.warning("Year %d: found %d duplicate timestamps, keeping first", year, n_dupes)
    df_year = df_year[~df_year.index.duplicated(keep="first")]

    return df_year if not df_year.empty else None


def fit_stats(df: pd.DataFrame) -> dict:
    """
    Compute mean and std for each column from a DataFrame (should be train set only).
    Returns stats_dict = { col: {"mean": float, "std": float}, ... }
    """
    stats = {}
    for col in df.columns:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }
    return stats


def apply_normalization(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Apply Z-score normalization using pre-computed stats (always from train set).
    Never fits on the data itself — prevents data leakage.
    """
    df_norm = df.copy()
    for col in df.columns:
        if col not in stats:
            logger.warning("Column '%s' not in stats, skipping normalization", col)
            continue
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        if std == 0:
            logger.warning("Column '%s' has std=0, skipping normalization", col)
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df[col] - mean) / std
    return df_norm


def clean(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Validate and clean a yearly DataFrame.
    Uses time-based interpolation to preserve temporal structure,
    with median fallback for edge cases (start/end of series).
    """
    nan_counts = df.isna().sum()
    if nan_counts.any():
        logger.warning("Year %d — NaNs found:\n%s", year, nan_counts[nan_counts > 0].to_string())
        df = df.interpolate(method="time")
        remaining = df.isna().sum().sum()
        if remaining:
            logger.warning("Year %d — %d NaNs remain after interpolation, filling with median", year, remaining)
            df = df.fillna(df.median(numeric_only=True))
    return df


def run_pipeline(years: list[int], normalize: bool = True) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_years = [y for y in years if y <= TRAIN_END]
    val_years   = [y for y in years if TRAIN_END < y <= VAL_END]
    test_years  = [y for y in years if y > VAL_END]

    logger.info("Split — train: %s, val: %s, test: %s",
                f"{min(train_years)}–{max(train_years)}" if train_years else "—",
                f"{min(val_years)}–{max(val_years)}"     if val_years   else "—",
                f"{min(test_years)}–{max(test_years)}"   if test_years  else "—")

    train_stats = None
    if normalize:
        logger.info("Fitting normalization stats on train set (%d–%d)...", min(train_years), max(train_years))
        train_frames = []
        for year in train_years:
            df = load_year(year)
            if df is not None:
                train_frames.append(clean(df, year))
        if train_frames:
            train_stats = fit_stats(pd.concat(train_frames))
            with open(STATS_FILE, "w") as f:
                json.dump(train_stats, f, indent=2)
            logger.info("Stats saved → %s", STATS_FILE)


    processed = 0
    for year in tqdm(years, desc="Years", unit="year"):
        out_path = PROCESSED_DATA_DIR / f"rome_{year}.parquet"

        logger.info("Processing %d...", year)
        df = load_year(year)

        if df is None or df.empty:
            logger.warning("No data for %d, skipping", year)
            continue

        df = clean(df, year)

        if normalize and train_stats:
            df = apply_normalization(df, train_stats)

        if year <= TRAIN_END:
            df["split"] = "train"
        elif year <= VAL_END:
            df["split"] = "val"
        else:
            df["split"] = "test"

        df.to_parquet(out_path, index=True)
        logger.info("Saved → %s | shape: %s", out_path, df.shape)
        processed += 1

    logger.info("Done. Processed %d year(s).", processed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERA5 NetCDF → Parquet pipeline")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(YEAR_START, YEAR_END + 1)),
        help="Years to process (default: 1996–2025)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip Z-score normalization and save raw values",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(years=args.years, normalize=not args.no_normalize)