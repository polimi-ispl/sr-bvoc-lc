import os
import xarray as xr
import fsspec
from tqdm import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool

# Open remote topdown SEEDS Isoprene dataset
ds_isoprene = xr.open_zarr(
  store=fsspec.get_mapper("https://data.seedsproject.eu/seeds_top-down-isoprene-emissions_bira-iasb_20180101-20221231_magritte_v2/slices.zarr"),
  chunks={'time': 'auto'}
)

ds_isoprene = ds_isoprene.rename({'isoprene_flux': 'isoprene'})

# Get the unique months in the dataset
unique_months = pd.date_range(start=ds_isoprene.time.min().values, end=ds_isoprene.time.max().values, freq='MS')

# Directory to save the files
output_dir = '<usr_dir>/data/SEEDS/top-down'

# Initialize the progress bar
with tqdm(total=len(unique_months), desc="Saving monthly data") as pbar:
    for month in unique_months:
        start_date = month.strftime('%Y-%m-%d')
        end_date = (month + pd.DateOffset(months=1) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        monthly_data = ds_isoprene.sel(time=slice(start_date, end_date))
        filename = f'{output_dir}/isoprene_{month.strftime("%Y_%m")}.nc'
        
        # Check if the file already exists
        if not os.path.exists(filename):
            monthly_data.to_netcdf(filename)
            print(f'Saved {filename}')
        else:
            print(f'File {filename} already exists')
        
        pbar.update(1)
