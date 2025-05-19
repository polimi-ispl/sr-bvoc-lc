import numpy as np
import rasterio
import xarray as xr
from tqdm import tqdm
import os
from multiprocessing import Pool

# Source for LCCS Class and Data
# WorldCover Product User Manual

def get_seeds_dataset_coords():
    # load target dataset coordinates
    file_path = '<usr_dir>/data/SEEDS/top-down/isoprene_2018_01.nc'

    # Open the dataset
    with xr.open_dataset(file_path) as ds:
        # Get the latitude and longitude coordinates
        lat = np.round(ds['latitude'].values, 2)
        lon = np.round(ds['longitude'].values, 2)
    return lat, lon

def resample_data(file_path, lat_target, lon_target):
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
        print('calculated transformed coordinates')
        print('lon:', lon)
        print('lat:', lat)

        # Create a DataArray with the geographical coordinates
        data_array = xr.DataArray(
            raster,
            dims=('latitude','longitude'),
            coords={
                'longitude': (('longitude',), lon),
                'latitude': (('latitude',), lat)
            }
        )

    data_array_binary = data_array.copy()
    # Create a binary cover map where pixels equal to 10 (Map Code for land cover class / LCCS: TREE COVER) are set to 1, and others to 0
    data_array_binary.data = np.where(data_array == 10, 1, 0).astype(np.uint8)

    min_lat = min(data_array_binary.latitude.values)
    max_lat = max(data_array_binary.latitude.values)
    min_lon = min(data_array_binary.longitude.values)
    max_lon = max(data_array_binary.longitude.values)

    print('lat_target range:', min(lat_target), max(lat_target))
    print('lon_target range:', min(lon_target), max(lon_target))

    subset_lat = [item for item in lat_target if min_lat <= item <= max_lat]
    subset_lon = [item for item in lon_target if min_lon <= item <= max_lon]

    if subset_lat == [] or subset_lon == []:
        print('No overlap between', file_path, 'and target dataset')
        return None
    else:
        print('Resampling', file_path, 'to', len(subset_lat), 'latitude and', len(subset_lon), 'longitude coordinates')
        print('Original size:', len(data_array_binary.latitude.values), 'latitude and', len(data_array_binary.longitude.values), 'longitude coordinates')
        print('min_lat:', min_lat, 'max_lat:', max_lat, 'min_lon:', min_lon, 'max_lon:', max_lon)

        lat_factor = len(data_array_binary.latitude.values) / len(subset_lat)
        lon_factor = len(data_array_binary.longitude.values) / len(subset_lon)

        # downsample by less than the full coarsening factor and then interpolation to target coordinates
        coarser_data_array = data_array_binary.coarsen(latitude=np.round(lat_factor * 0.75, -2).astype(int) , longitude=np.round(lon_factor * 0.75, -2).astype(int), boundary='trim').mean()
        resampled_data_array = coarser_data_array.interp(latitude=subset_lat, longitude=subset_lon)

        result = resampled_data_array.rename('forest_percentage')

        return result

data_dir = '<usr_dir>/data/worldcover/original'

lat_target, lon_target = get_seeds_dataset_coords()

results = None

# Iterate over all files in the directory
for root, dirs, files in os.walk(data_dir):
    print('Processing', len(files), 'files in', root)
    for i, file in enumerate(tqdm(files)):
        file_path = os.path.join(root, file)

        result = resample_data(file_path, lat_target, lon_target)

        if results is None:
            results = resample_data(file_path, lat_target, lon_target)
        else:
            if result is not None:
                results = xr.merge([results, result])

destination_file = '<usr_dir>/data/worldcover/masks/percentage_forest.nc'

print('Size of results:', results.nbytes / 1e6, 'MB ...')
print('Saving results to', destination_file)
results.to_netcdf(destination_file)
