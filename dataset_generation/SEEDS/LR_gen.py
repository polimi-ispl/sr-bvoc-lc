import os
import numpy as np
import cv2
import pandas as pd
import xarray as xr
from multiprocessing import Pool
from tqdm import tqdm

def bicubic_isoprene_LR(isoprene_dir, ag_dir, forest_dir, ds_ag, ds_forest, dataset_index, sr_factor, day):

    day_patches = dataset_index.loc[dataset_index['day'] == day]

    filepaths = []

    for index, row in day_patches.iterrows():
        patch_coords = row['patch']
        HR_file_path = row['HR_file_path']
        HR_patch_data = np.load(HR_file_path)

        patch_coords = eval(patch_coords)
        ag_patch = ds_ag.sel(latitude=slice(patch_coords['lat'][0], patch_coords['lat'][1]), longitude=slice(patch_coords['lon'][0], patch_coords['lon'][1]))
        forest_patch = ds_forest.sel(latitude=slice(patch_coords['lat'][0], patch_coords['lat'][1]), longitude=slice(patch_coords['lon'][0], patch_coords['lon'][1]))
        # print("ag_patch shape:", ag_patch['ag_percentage'].shape)
        # print("forest_patch shape:", forest_patch['forest_percentage'].shape)

        if HR_patch_data.shape[1] != HR_patch_data.shape[0]:
            print("HR patch data shape is not square!")
            break

        if HR_patch_data.shape[1] % sr_factor != 0:
            print("HR patch data shape is not divisible by the scale factor!")
            break

        if (HR_patch_data.shape[1] != ag_patch['ag_percentage'].shape[1]) or (HR_patch_data.shape[0] != ag_patch['ag_percentage'].shape[0]):
            print("patch shapes do not match!")
            break
        if (HR_patch_data.shape[1] != forest_patch['forest_percentage'].shape[1]) or (HR_patch_data.shape[0] != forest_patch['forest_percentage'].shape[0]):
            print("patch shapes do not match!")
            break

        LR_isoprene_patch = cv2.resize(HR_patch_data, (HR_patch_data.shape[1] // sr_factor, HR_patch_data.shape[0] // sr_factor), interpolation=cv2.INTER_CUBIC)
        # print("downsampled HR patch data shape:", LR_isoprene_patch.shape)
        LR_isoprene_patch[LR_isoprene_patch < 0] = 0 # no negative emissions
        
        LR_ag_patch = cv2.resize(ag_patch['ag_percentage'].values, (ag_patch['ag_percentage'].shape[1] // sr_factor, ag_patch['ag_percentage'].shape[0] // sr_factor), interpolation=cv2.INTER_CUBIC)
        # print("downsampled ag patch shape:", LR_ag_patch.shape)
        LR_ag_patch[LR_ag_patch < 0] = 0 # no negative agriculture percentage

        LR_forest_patch = cv2.resize(forest_patch['forest_percentage'].values, (forest_patch['forest_percentage'].shape[1] // sr_factor, forest_patch['forest_percentage'].shape[0] // sr_factor), interpolation=cv2.INTER_CUBIC)
        # print("downsampled forest patch shape:", LR_forest_patch.shape)
        LR_forest_patch[LR_forest_patch < 0] = 0 # no negative forest percentage

        day_str = pd.to_datetime(day).strftime('%Y_%m_%d')

        file_name = f'LR_patch_{day_str}_lon{str(patch_coords["lon"][0]).replace(".", "")}_lat{str(patch_coords["lat"][0]).replace(".", "")}.npy'
        isoprene_file_path = os.path.join(isoprene_dir, file_name)
        np.save(isoprene_file_path, LR_isoprene_patch)

        file_name = f'AG_patch_{day_str}_lon{str(patch_coords["lon"][0]).replace(".", "")}_lat{str(patch_coords["lat"][0]).replace(".", "")}.npy'
        ag_file_path = os.path.join(ag_dir, file_name)
        np.save(ag_file_path, LR_ag_patch)

        file_name = f'FOREST_patch_{day_str}_lon{str(patch_coords["lon"][0]).replace(".", "")}_lat{str(patch_coords["lat"][0]).replace(".", "")}.npy'
        forest_file_path = os.path.join(forest_dir, file_name)
        np.save(forest_file_path, LR_forest_patch)

        filepaths.append((index, isoprene_file_path, ag_file_path, forest_file_path))

    return filepaths

dataset_dir = '<usr_dir>/datasets/p30pix_s1deg_z10_isoprene_ag_forest'
sr_factor = 2

dataset_file_path = os.path.join(dataset_dir, 'dataset_index.csv')
dataset_index = pd.read_csv(dataset_file_path, index_col=0)
print("dataset_index shape", dataset_index.shape)

ag_file_path = '<usr_dir>/data/worldcover/masks/percentage_agriculture.nc'
ds_ag = xr.open_dataset(ag_file_path)
print("ds_ag shape", ds_ag['ag_percentage'].shape)

forest_file_path = '<usr_dir>/data/worldcover/masks/percentage_forest.nc'
ds_forest = xr.open_dataset(forest_file_path)
print("ds_forest shape", ds_forest['forest_percentage'].shape)

days = dataset_index["day"].unique()
print(f"Number of days: {len(days)}")

HR_data_dir = os.path.join(dataset_dir, "HR")
LR_data_dir = os.path.join(dataset_dir, "LR"+"_sr"+str(sr_factor))

if not os.path.exists(LR_data_dir):
    os.makedirs(LR_data_dir)

isoprene_dir = os.path.join(LR_data_dir, "isoprene")
semantic_dir = os.path.join(LR_data_dir, "semantic")
ag_dir = os.path.join(semantic_dir, "ag")
forest_dir = os.path.join(semantic_dir, "forest")

if not os.path.exists(isoprene_dir):
    os.makedirs(isoprene_dir)
if not os.path.exists(ag_dir):
    os.makedirs(ag_dir)
if not os.path.exists(forest_dir):
    os.makedirs(forest_dir)

with Pool() as pool:

    with tqdm(total=len(days)) as pbar:
        def update(*a):
            pbar.update()

        # too big if not working on a single day
        results = [pool.apply_async(bicubic_isoprene_LR,
                                    args=(isoprene_dir, ag_dir, forest_dir, ds_ag, ds_forest, dataset_index, sr_factor, day),
                                    callback=update) for day in days]
        # Wait for all processes to finish
        for res in results:
            res.get()

    # Ensure the progress bar is closed
    pbar.close()

    # add empty columns to dataset_index
    dataset_index["LR_file_path"] = ""
    dataset_index["AG_file_path"] = ""
    dataset_index["FOREST_file_path"] = ""
    # Update dataset_index with LR_file_path column
    for res in tqdm(results):
        file_paths = res.get()
        for index, LR_file_path, ag_file_path, forest_file_path in file_paths:
            dataset_index.loc[index, "LR_file_path"] = LR_file_path
            dataset_index.loc[index, "AG_file_path"] = ag_file_path
            dataset_index.loc[index, "FOREST_file_path"] = forest_file_path

# Save the dataset_index
dataset_index.to_csv(f"{dataset_dir}/dataset_index.csv")
