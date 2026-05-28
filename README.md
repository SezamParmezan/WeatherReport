# WeatherReport

WeatherReport is a data-driven weather modeling project focused on preparing, processing, and analyzing ERA5 climate reanalysis data for a specific geographic region around Rome, Italy.

## 🌤️ What this project does
- Downloads and organizes monthly ERA5 netCDF weather data.
- Processes raw atmospheric and surface variables for modeling.
- Builds a clean dataset ready for analysis or machine learning.

## 📁 Repository structure
- `LOAD_DATA.py` — download helper for ERA5 single-level data from the Copernicus Climate Data Store.
- `data/process.py` — data cleaning and preprocessing logic.
- `data/raw/era5/` — stored raw downloaded ERA5 `.nc` files.
- `data/processed/` — output location for processed data artifacts.
- `data_vars.md` — notes on dataset variables and metadata.

## 🚀 Getting started
1. Install dependencies for your Python environment.
2. Create a CDS API key file named `.cdsapirc`.
   - Windows: `C:/Users/<your_user>/.cdsapirc`
   - macOS/Linux: `/Users/<your_user>/.cdsapirc` or `/home/<your_user>/.cdsapirc`
3. Run the downloader:
   ```bash
   python LOAD_DATA.py
   ```
4. Process the raw data:
   ```bash
   python data/process.py
   ```

## 📌 Notes
- The downloader is configured to export ERA5 monthly files for Rome from 1996 onward.
- The area boundary and variables can be adjusted in `LOAD_DATA.py`.
- Processed outputs live in `data/processed/`.

## 🧠 Data source
This project uses ERA5 reanalysis data from the Copernicus Climate Data Store, including variables such as temperature, pressure, wind, humidity, precipitation, and cloud cover.

## ✅ Goal
Create a reusable weather dataset pipeline for predictive modeling in a localized region, with clear steps for downloading, organizing, and preprocessing ERA5 weather data.
