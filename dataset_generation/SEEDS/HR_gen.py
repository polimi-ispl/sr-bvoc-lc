import os
import xarray as xr
import fsspec
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import sys
sys.path.append('<usr_dir>/dataset_generation')

def generate_index(ds_isoprene, patch_coords):
    """
    Generate a MultiIndex DataFrame with a numerical index, a day index, 
    and for each day all the patches defined by patch_coords.
    
    Parameters:
    - ds_isoprene: Xarray Dataset containing the 'time' dimension.
    - patch_coords: List of dictionaries, where each dictionary contains 'lat' and 'lon' lists.
    
    Returns:
    - df: Pandas DataFrame with a MultiIndex.
    """
    # Extract days
    days = ds_isoprene.time.values
    
    data = []
    for day in days:
        for patch in patch_coords:
            data.append((day, patch))
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["day", "patch"])

    return df


def extract_isoprene_HR(data_dir, dataset_dir, dataset_index, zero_threshold, day):
    filepaths = []

    # Find the right file based on the day    
    date_str = pd.to_datetime(day).strftime('%Y_%m')
    file_name = f'isoprene_{date_str}.nc'
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return None
    
    # Load the dataset
    ds_isoprene = xr.open_dataset(file_path)

    # select the day
    day_isoprene = ds_isoprene.sel(time=day)

    # Select all rows in dataset_index with the given day
    rows_with_day = dataset_index[dataset_index["day"] == day]

    for index, row in rows_with_day.iterrows():

        patch_coords = row["patch"]
        # Select the patch
        patch = day_isoprene.sel(latitude=slice(patch_coords['lat'][0], patch_coords['lat'][1]), longitude=slice(patch_coords['lon'][0], patch_coords['lon'][1]))
        # print("patch selected")
        
        # Compute the zero percentage
        zero_percentage_isoprene = np.count_nonzero(patch['isoprene'] == 0) / patch['isoprene'].size * 100
        # print("zero percentage computed")
        
        if zero_percentage_isoprene < zero_threshold:
            # Convert the patch to a NumPy array
            patch_np = patch['isoprene'].values
            # Save patch_np to dataset_dir
            day_str = pd.to_datetime(day).strftime('%Y_%m_%d')
            file_name = f'HR_patch_{day_str}_lon{str(patch_coords["lon"][0]).replace(".", "")}_lat{str(patch_coords["lat"][0]).replace(".", "")}.npy'
            file_path = os.path.join(dataset_dir, file_name)
            np.save(file_path, patch_np)
            # print(f"Patch saved to {file_path}")
        else:
            file_path = None

        filepaths.append((index, file_path))

    return filepaths


# Open remote topdown SEEDS Isoprene dataset
ds_isoprene = xr.open_zarr(
  store=fsspec.get_mapper("https://data.seedsproject.eu/seeds_top-down-isoprene-emissions_bira-iasb_20180101-20221231_magritte_v2/slices.zarr"),
  chunks={'time': 'auto'}
)

ds_isoprene = ds_isoprene.rename({'isoprene_flux': 'isoprene'})

# Determine the step size based on the number of pixels
pixel_size = 0.1  # each pixel is 0.1 degree by 0.1 degree
num_pixels = 30  # number of pixels desired in each patch to get 3 degrees
increment = 1  # stride

patch_size = (num_pixels - 1) * pixel_size

# Select the latitude range
min_latitude = ds_isoprene.latitude.min().values.astype(np.float64)
max_latitude = ds_isoprene.latitude.max().values.astype(np.float64)

# Select the longitude range
min_longitude = ds_isoprene.longitude.min().values.astype(np.float64)
max_longitude = ds_isoprene.longitude.max().values.astype(np.float64)


patch_coords = []
current_latitude = min_latitude + 0
while current_latitude + patch_size <= max_latitude:
    current_longitude = min_longitude + 0
    while current_longitude + patch_size <= max_longitude:
        patch_coords.append({
            "lat": [round(current_latitude, 2), round(current_latitude + patch_size, 2)],
            "lon": [round(current_longitude, 2), round(current_longitude + patch_size, 2)]
        })
        current_longitude = current_longitude + increment
    current_latitude = current_latitude + increment

print("number of patch_coords", len(patch_coords))
print("number of days", len(ds_isoprene.time.values))

dataset_index = generate_index(ds_isoprene, patch_coords)
print("length of dataset_index", len(dataset_index))

data_dir = "<usr_dir>/data/SEEDS/top-down"
zero_threshold = 10 # %
dataset_dir = f"<usr_dir>/datasets/p{num_pixels}pix_s{increment}deg_z{zero_threshold}_isoprene_ag_forest"
HR_path = os.path.join(dataset_dir, "HR")


# Create the directory if it doesn't exist
if not os.path.exists(HR_path):
    os.makedirs(HR_path)


with Pool() as pool:
    days = dataset_index["day"].unique()

    with tqdm(total=len(days)) as pbar:
        def update(*a):
            pbar.update()

        # too big if not working on a single day
        results = [pool.apply_async(extract_isoprene_HR,
                                    args=(data_dir,
                                          HR_path,
                                          dataset_index,
                                          zero_threshold,
                                          day),
                                    callback=update) for day in days]
        
        # Wait for all processes to finish
        for res in results:
            res.get()

    # Ensure the progress bar is closed
    pbar.close()

    # add HR_file_path to dataset_index
    dataset_index["HR_file_path"] = ""
    # Update dataset_index with HR_file_path column
    for res in tqdm(results):
        file_paths = res.get()
        for index, file_path in file_paths:
            dataset_index.loc[index, "HR_file_path"] = file_path

dataset_index.dropna(subset=['HR_file_path'], inplace=True)
dataset_index.reset_index(drop=True, inplace=True)
# Save the dataset_index
dataset_index.to_csv(f"{dataset_dir}/dataset_index.csv")
