import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.plot import show
from rasterio.enums import Resampling
import rasterio.warp as warp
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import os

def get_seeds_dataset_coords():
    # load target dataset coordinates
    file_path = '<usr_dir>/data/SEEDS/top-down/isoprene_2018_01.nc'

    # Open the dataset
    with xr.open_dataset(file_path) as ds:
        # Get the latitude and longitude coordinates
        lat = np.round(ds['latitude'].values, 2)
        lon = np.round(ds['longitude'].values, 2)
    return lat, lon

data_dir = '<usr_dir>/data/worldcover/original'

lat_target, lon_target = get_seeds_dataset_coords()

# Iterate over all files in the directory
for root, dirs, files in os.walk(data_dir):
    for i, file in enumerate(tqdm(files)):

        file_path = os.path.join(root, file)
        # Open the GeoTIFF file
        with rasterio.open(file_path) as src:
            # Read the raster data
            raster = src.read(1) # read the first (only) band; returns a Numpy N-D array

            # Get the pixel coordinates 
            x, y = raster.shape
            x = np.arange(x)
            y = np.arange(y)

            # Convert the pixel coordinates to geographical coordinates
            lon, lat = rasterio.transform.xy(src.transform, x, y)

            # Create a DataArray with the geographical coordinates
            data_array = xr.DataArray(
                raster,
                dims=('longitude','latitude'),
                coords={
                    'longitude': (('longitude',), lon),
                    'latitude': (('latitude',), lat)
                }
            )

        data_array_binary = data_array.copy()
        # Create a binary cover map where pixels equal to 40 (Map Code for land cover class / LCCS: CROPLAND) are set to 1, and others to 0
        data_array_binary.data = np.where(data_array == 40, 1, 0).astype(np.uint8)

        min_lat = min(data_array_binary.latitude.values)
        max_lat = max(data_array_binary.latitude.values)
        min_lon = min(data_array_binary.longitude.values)
        max_lon = max(data_array_binary.longitude.values)

        subset_lat = [item for item in lat_target if min_lat <= item <= max_lat]
        subset_lon = [item for item in lon_target if min_lon <= item <= max_lon]

        lat_factor = len(data_array_binary.latitude.values) / len(subset_lat)
        lon_factor = len(data_array_binary.longitude.values) / len(subset_lon)

        # downsample by less than the full coarsening factor and then interpolation to target coordinates
        coarser_data_array = data_array_binary.coarsen(latitude=np.round(lat_factor * 0.75, -2).astype(int) , longitude=np.round(lon_factor * 0.75, -2).astype(int), boundary='trim').mean()
        resampled_data_array = coarser_data_array.interp(latitude=subset_lat, longitude=subset_lon)

        # save results
        if i == 0:
            results = resampled_data_array.rename('ag_percentage')
        else:
            results = xr.merge([results, resampled_data_array.rename('ag_percentage')])

print('Size of results:', results.nbytes / 1e6, 'MB ...')

results.to_netcdf('<usr_dir>/data/worldcover/masks/percentage_agriculture.nc')
