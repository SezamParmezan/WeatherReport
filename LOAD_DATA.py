import os
import cdsapi
import shutil
import zipfile

from pathlib import Path

"""
This script downloads ERA5 single-level data from the CDS API and saves it to
"data/raw/era5/rome_<year>_<month>.nc".

Before running:
- Create a CDS API key file named `.cdsapirc`.
- On Windows, place it in `C:/Users/<your_user>/.cdsapirc`.
- On macOS or Linux, place it in `/Users/<your_user>/.cdsapirc` or `/home/<your_user>/.cdsapirc`.
- Copy the exact credentials from the CDS website into that file.

Important parts to change for custom downloads:
- year: the range of years in the outer loop. Change range(1996, 2026) if you
  want a different period.
- month: the inner loop over months. Change range(1, 13) to download fewer months.
- request: the dictionary defines variables, days, times, area, and file format.
  Change the variables list or area bounds here if needed.

How it works:
- The script creates the output folder if it does not exist.
- It loops over each year and month, skipping any monthly file already downloaded.
- It calls CDS API with one year/month request and saves each month to a separate
  netCDF file.
"""

os.makedirs("data/raw/era5", exist_ok=True)

client = cdsapi.Client()

track = 0

for year in range(1996, 2026):
    for month in range(1, 13):
        output_path = f"data/raw/era5/rome_{year}_{month:02d}.nc"

        if os.path.exists(output_path):
            print(f"[skip] {year}-{month:02d} already exists, skipping")
            continue

        track += 1
        print(f"[{track}] Downloading {year}-{month:02d}...")

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "mean_sea_level_pressure",
                "surface_pressure",
                "total_precipitation",
                "total_cloud_cover",
            ],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [42.022548, 12.311565, 41.761946, 12.661647],
        }

        client.retrieve("reanalysis-era5-single-levels", request).download(output_path)
        print(f"[{track}] {year}-{month:02d} saved in {output_path}")

track += 1
print(f"\n[{track}] Organizing and extracting files...")

for nc_path in Path("data/raw/era5").glob("rome_*.nc"):
    year = nc_path.name.split("_")[1]
    year_dir = Path(f"data/raw/era5/{year}")
    year_dir.mkdir(exist_ok=True)

    dest = year_dir / nc_path.name

    with open(nc_path, "rb") as f:
        magic = f.read(2)

    track += 1
    if magic == b"PK":
        with zipfile.ZipFile(nc_path) as zf:
            zf.extractall(year_dir)
        nc_path.unlink()
        print(f"[{track}] {nc_path.name} extracted to {year_dir}")
    else:
        shutil.move(str(nc_path), str(dest))
        print(f"[{track}] {nc_path.name} moved to {year_dir}")


      
print(f"\n[{track + 1}] All files organized in data/raw/era5/<year>/rome_<year>_<month>.nc")