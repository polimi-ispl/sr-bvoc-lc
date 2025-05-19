import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from datetime import datetime
import argparse
import geopandas as gpd
from shapely.geometry import Polygon, Point

def parse_args():
    parser = argparse.ArgumentParser(description="Partition generation for climate dataset")
    parser.add_argument('--train_samples', type=int, default=75000, help='Percentage of training samples')
    parser.add_argument('--val_samples', type=int, default=5000, help='Percentage of validation samples')
    # parser.add_argument('--test', type=int, default=15, help='Percentage of test samples')
    parser.add_argument('--samples', type=int, default=100000, help='Number of samples')
    parser.add_argument('--climate_cv_size', type=int, default=10000, help='Number of climate cross-validation samples')
    parser.add_argument('--spatial_cv_size', type=int, default=10000, help='Minimum number of spatial cross-validation samples')
    parser.add_argument('--partition_tag', type=str, default='_', help='Tag for the partitioned dataset index file')
    parser.add_argument('--exclude_list', type=int, nargs='+', default=[0], help='List of climate classes to exclude for training')
    parser.add_argument('--include_list', type=int, nargs='+', default=[0], help='List of climate classes to include in spatial cross-validation')
    parser.add_argument('--partition_mode', type=str, default='random', choices=['random', 'sequential'], help='Partition mode')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--holdout_point', type=int, nargs='+', default=[0], help='List of climate classes to exclude for training')
    return parser.parse_args()

args = parse_args()

partition_logic = {
    'train': args.train_samples,
    'val': args.val_samples,
    # 'test': args.test,
    'samples': args.samples,
    'spatial_cv_size': args.spatial_cv_size,
    'climate_cv_size': args.climate_cv_size
}

holdout_point = args.holdout_point
partition_tag = args.partition_tag
exclude_list = args.exclude_list
include_list = args.include_list
partition_mode = args.partition_mode
seed = args.seed
dataset_path = args.dataset_path

print("loading dataset index...")

climate_index_file = os.path.join(dataset_path, 'dataset_index_climate.csv')
if os.path.exists(climate_index_file):
    print(f"Loading existing climate index file from {climate_index_file}...")
    df = pd.read_csv(climate_index_file, index_col=0)
else:
    print("Climate index file not found. Proceeding with the generation...")

    df = pd.read_csv(os.path.join(dataset_path, 'dataset_index.csv'), index_col=0)

    # Load the Climate Zone NetCDF file
    koppen_geiger_file_path = '<usr_dir>/data/koppen_geiger/koppen_geiger_0p1_SEEDS_region.nc'
    ds_koppen_geiger = xr.open_dataset(koppen_geiger_file_path)

    df['climate_class'] = None  # Initialize the 'climate_class' column with empty lists
    print("Assigning climate classes to each patch...")
    df['patch'] = df['patch'].apply(eval)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        patch = row['patch']
        lat_min, lat_max = patch['lat']
        lon_min, lon_max = patch['lon']
        koppen_patch = ds_koppen_geiger.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        df.at[index, 'climate_class'] = list(np.unique(koppen_patch.climate_code.values))

    num_nones = df['climate_class'].isna().sum()
    print(f"Number of Nones in 'climate_class' column: {num_nones}")

    empty_climate_class = df['climate_class'].apply(lambda x: len(x) == 0).sum()
    print(f"Number of rows with empty 'climate_class' lists: {empty_climate_class}")

    df.to_csv(climate_index_file)

# evaluate the climate classes and patch coordinates
df['climate_class'] = df['climate_class'].apply(eval)
df['patch'] = df['patch'].apply(eval)

dataset_partition_index = df.copy()

dataset_partition_index['geosplit'] = None

ds_index = dataset_partition_index.index.tolist()

rng = np.random.default_rng(seed=seed)

# Split the dataset geographically for spatial cross-validation by selecting climate classes
print('Splitting the dataset geographically for spatial cross-validation...')

# Create the train / val / test split excluding the cross-validation climate classes
partition_filtered_df = dataset_partition_index[~dataset_partition_index['climate_class'].apply(lambda x: any(climate in exclude_list for climate in x))]
dataset_partition_index.loc[partition_filtered_df.index, 'geosplit'] = 1
print(f"Number of samples in the partitioned dataset: {partition_filtered_df.shape[0]}")

# Create the climate cross-validation split
climate_cv_filtered_df = dataset_partition_index[dataset_partition_index['climate_class'].apply(lambda x: all(climate in include_list for climate in x))]
print(f"Number of samples in climate cross-validation set: {climate_cv_filtered_df.shape[0]}")
climate_cv_size = partition_logic['climate_cv_size']
climate_cv_index = climate_cv_filtered_df.index.tolist() # TODO make sampling part of the dataset for the test / climate_cv / spatial_cv scripts
dataset_partition_index.loc[climate_cv_index, 'geosplit'] = 'climate'

# Create the spatial cross-validation split
filtered_df = dataset_partition_index[dataset_partition_index['geosplit'] == 1]

# Function to create a Polygon from patch bounds
def create_polygon(patch):
    lat_min, lat_max = patch['lat']
    lon_min, lon_max = patch['lon']
    # Create a Polygon using the corner points of the patch
    return Polygon([(lon_min, lat_min), (lon_max, lat_min), (lon_max, lat_max), (lon_min, lat_max)])

# Create a GeoDataFrame with the patches as polygons
filtered_df['geometry'] = filtered_df['patch'].apply(create_polygon)
gdf = gpd.GeoDataFrame(filtered_df, geometry='geometry')

# Find the geometric center of filtered patches
center = gdf.union_all().centroid
center_coords = (center.x, center.y)
centroid_point = Point(center_coords)

def patch_centroid_distance(patch_polygon, centroid_point):
    patch_centroid = patch_polygon.centroid
    return patch_centroid.distance(centroid_point)

# force the centroid to be the holdout_point input if it exists
if len(holdout_point) == 2:
    longitude = holdout_point[0]
    latitude = holdout_point[1]
    # Create a Point object with the specified coordinates
    centroid_point = Point(longitude, latitude)
    print(f"Centroid set to input: {centroid_point}")

# Compute the distance for each patch and sort by distance
gdf['distance_to_centroid'] = gdf['geometry'].apply(lambda x: patch_centroid_distance(x, centroid_point))
gdf_sorted = gdf.sort_values(by='distance_to_centroid')

selected_patches = gdf_sorted.head(partition_logic['spatial_cv_size'])

# Assign the spatial cross-validation split
spatial_cv_index = selected_patches.index.tolist()
dataset_partition_index.loc[spatial_cv_index, 'geosplit'] = 'spatial'

# Create a boolean mask where patches do NOT intersect with the selected region
selected_patches_region = selected_patches['geometry'].union_all()
non_intersecting_patches = gdf[~gdf['geometry'].intersects(selected_patches_region)]

if len(non_intersecting_patches) < partition_logic['samples']:
    raise ValueError(f"Number of non-intersecting patches ({len(non_intersecting_patches)}) is less than {partition_logic['samples']}")

# Assign the partition selection which is not in the spatial cross-validation split
non_intersecting_partition_index = non_intersecting_patches.index.tolist()
dataset_partition_index.loc[non_intersecting_partition_index, 'geosplit'] = 'partition'

# Partition to train, val, test
filtered_dataset_partition_index = dataset_partition_index[dataset_partition_index['geosplit'] == 'partition']
# sample_size = partition_logic['samples']
# random_sample = filtered_dataset_partition_index.sample(n=sample_size, random_state=rng)

# Partition the dataset into train, validation, and test sets
print("Partitioning the dataset...")
train_size = partition_logic['train']
val_size = partition_logic['val']
# test_percentage = partition_logic['test']

# train_size = int(sample_size * train_percentage / 100)
# val_size = int(sample_size * val_percentage / 100)

random_index = filtered_dataset_partition_index.index.tolist()
rng.shuffle(random_index)
train_index = random_index[:train_size]
val_index = random_index[train_size:train_size + val_size]
test_index = random_index[train_size + val_size:]

dataset_partition_index['partition'] = None
dataset_partition_index.loc[train_index, 'partition'] = 'train'
dataset_partition_index.loc[val_index, 'partition'] = 'val'
dataset_partition_index.loc[test_index, 'partition'] = 'test'

print("train samples:", dataset_partition_index[dataset_partition_index['partition'] == 'train'].shape[0])
print("val samples:", dataset_partition_index[dataset_partition_index['partition'] == 'val'].shape[0])
print("test samples:", dataset_partition_index[dataset_partition_index['partition'] == 'test'].shape[0])

# Save the partitioned dataset index
current_time = datetime.now().strftime("%Y%m%d_%H%M")
filename = f'climate{partition_tag}_partition_index_t{len(train_index)}_v{len(val_index)}_te{len(test_index)}_scv{partition_logic['spatial_cv_size']}_seed{seed}_{current_time}.csv'
print(f"Saving partitioned dataset index to {filename}")
dataset_partition_index.to_csv(os.path.join(dataset_path, filename))
