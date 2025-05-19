import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from pickle import dump
from time import time
from joblib import parallel_backend
from multiprocessing import Pool

def load_and_process(file_tuple):
    """
    Load and process the HR and LR arrays from the given file tuple.
    """
    HR_file_path, LR_file_path = file_tuple
    HR_arr = np.load(HR_file_path)
    LR_arr = np.load(LR_file_path)
    return HR_arr, LR_arr

def compute_quantile_transformation(root, files, n_quantiles = 1000, flag_name=''):
    # Set the max number of spawned threads
    os.environ['OMP_NUM_THREADS'] = str(4)
    os.environ['OPENBLAS_NUM_THREADS'] = str(4)  # if PIP was used

    DATA_PATH = root
    print(f'[QT] Data Path: \t{DATA_PATH}')

    # Use multiprocessing to parallelize loading and processing
    print(f'[QT] Loading data ...')
    with Pool() as pool:
        results = list(tqdm(pool.imap(load_and_process, [(HR_file_path, LR_file_path) for HR_file_path, LR_file_path, _1, _2 in files]), total=len(files)))

    # Process results
    stacked_HR_arrays = []
    stacked_LR_arrays = []
    for HR_arr, LR_arr in results:
        stacked_HR_arrays.append(HR_arr)
        stacked_LR_arrays.append(LR_arr)

    # Stack arrays
    stacked_HR_arrays = np.dstack(stacked_HR_arrays)
    stacked_LR_arrays = np.dstack(stacked_LR_arrays)

    print(f'[QT] Start quantile transformations calculation ...')

    # Reshaping
    shape_HR = stacked_HR_arrays.shape
    shape_LR = stacked_LR_arrays.shape
    num_examples_HR = int(shape_HR[0] * shape_HR[1] * shape_HR[2])
    num_examples_LR = int(shape_LR[0] * shape_LR[1] * shape_LR[2])

    HR_data_flatten = stacked_HR_arrays.reshape(-1, 1)  # (num_examples_unrolled, 1)
    LR_data_flatten = stacked_LR_arrays.reshape(-1, 1)  # (num_examples_unrolled, 1)

    # Multiple test on transformer params HR
    assert num_examples_HR > n_quantiles, f'The "subsample" field is set to {num_examples_HR} and has to be > than "n_quantiles" field {n_quantiles}'
    with parallel_backend('threading', n_jobs=8):
        # -------- transformer HR
        quantile_transformer_HR_1e3qua_fullsub = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles,
                                                                     subsample=num_examples_HR,
                                                                     random_state=10).fit(HR_data_flatten)
        dump(quantile_transformer_HR_1e3qua_fullsub,
             open(os.path.join(DATA_PATH, f'quantile_transformer_HR_1e3qua_fullsub_{flag_name}.pkl'), 'wb'))
        print(f'HR Done <--')

        # -------- transformer LR
        quantile_transformer_LR_1e3qua_fullsub = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles,
                                                                     subsample=num_examples_LR,
                                                                     random_state=10).fit(LR_data_flatten)
        dump(quantile_transformer_LR_1e3qua_fullsub,
             open(os.path.join(DATA_PATH, f'quantile_transformer_LR_1e3qua_fullsub_{flag_name}.pkl'), 'wb'))
        print(f'LR Done <--')

    return quantile_transformer_HR_1e3qua_fullsub, quantile_transformer_LR_1e3qua_fullsub

def compute_log_quantile_transformation(root, files, n_quantiles = 1000, flag_name=''):

    # Set the max number of spawned threads
    os.environ['OMP_NUM_THREADS'] = str(4)
    os.environ['OPENBLAS_NUM_THREADS'] = str(4)  # if PIP was used

    DATA_PATH = root
    print(f'[QT] Data Path: \t{DATA_PATH}')

    # Use multiprocessing to parallelize loading and processing
    print(f'[QT] Loading data ...')
    with Pool() as pool:
        results = list(tqdm(pool.imap(load_and_process, [(HR_file_path, LR_file_path) for HR_file_path, LR_file_path, _1, _2 in files]), total=len(files)))

    # Process results
    stacked_HR_arrays = []
    stacked_LR_arrays = []
    global_log_min = np.abs(-35.068836)+1e-6
    for HR_arr, LR_arr in results:

        global_log_min = np.abs(-35.068836)+1e-6
        # Log transformation for non-zero values
        # Create masks for positive values
        hr_mask = HR_arr > 0
        lr_mask = LR_arr > 0

        # Apply log transformation only to positive values
        hr_data_l = np.zeros(HR_arr.shape)
        lr_data_l = np.zeros(LR_arr.shape)

        hr_data_l[hr_mask] = np.log(HR_arr[hr_mask]) + global_log_min
        lr_data_l[lr_mask] = np.log(LR_arr[lr_mask]) + global_log_min

        # HR_arr = np.log1p(HR_arr)
        # LR_arr = np.log1p(LR_arr)
        
        stacked_HR_arrays.append(hr_data_l)
        stacked_LR_arrays.append(lr_data_l)

    # Stack arrays
    stacked_HR_arrays = np.dstack(stacked_HR_arrays)
    stacked_LR_arrays = np.dstack(stacked_LR_arrays)

    print(f'[QT] Start quantile transformations calculation ...')

    # Reshaping
    shape_HR = stacked_HR_arrays.shape
    shape_LR = stacked_LR_arrays.shape
    num_examples_HR = int(shape_HR[0] * shape_HR[1] * shape_HR[2])
    num_examples_LR = int(shape_LR[0] * shape_LR[1] * shape_LR[2])

    HR_data_flatten = stacked_HR_arrays.reshape(-1, 1)  # (num_examples_unrolled, 1)
    LR_data_flatten = stacked_LR_arrays.reshape(-1, 1)  # (num_examples_unrolled, 1)

    # Multiple test on transformer params HR
    assert num_examples_HR > n_quantiles, f'The "subsample" field is set to {num_examples_HR} and has to be > than "n_quantiles" field {n_quantiles}'
    with parallel_backend('threading', n_jobs=8):
        # -------- transformer HR
        quantile_transformer_HR_1e3qua_fullsub = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles,
                                                                     subsample=num_examples_HR,
                                                                     random_state=10).fit(HR_data_flatten)
        dump(quantile_transformer_HR_1e3qua_fullsub,
             open(os.path.join(DATA_PATH, f'log_quantile_transformer_HR_1e3qua_fullsub_{flag_name}.pkl'), 'wb'))
        print(f'HR Done <--')

        # -------- transformer LR
        quantile_transformer_LR_1e3qua_fullsub = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles,
                                                                     subsample=num_examples_LR,
                                                                     random_state=10).fit(LR_data_flatten)
        dump(quantile_transformer_LR_1e3qua_fullsub,
             open(os.path.join(DATA_PATH, f'log_quantile_transformer_LR_1e3qua_fullsub_{flag_name}.pkl'), 'wb'))
        print(f'LR Done <--')

    return quantile_transformer_HR_1e3qua_fullsub, quantile_transformer_LR_1e3qua_fullsub

def calculate_patches_and_leftovers(H, W, P_h, P_w, S_h, S_w):
    """
    Calculate the number of patches that fit in an image and the leftover pixels in both dimensions.

    Parameters:
    H (int): Height of the image.
    W (int): Width of the image.
    P_h (int): Height of each patch.
    P_w (int): Width of each patch.
    S_h (int): Stride in the vertical direction (height).
    S_w (int): Stride in the horizontal direction (width).

    Returns:
    tuple: A tuple containing:
        - num_patches_h (int): Number of full patches that fit within the image height.
        - covered_height (int): Total height covered by the patches.
        - leftover_height (int): Number of pixels left after placing the full patches in the height dimension.
        - num_patches_w (int): Number of full patches that fit within the image width.
        - covered_width (int): Total width covered by the patches.
        - leftover_width (int): Number of pixels left after placing the full patches in the width dimension.
    """
    # Number of patches that can fit in height and width
    num_patches_h = (H - P_h) // S_h + 1
    num_patches_w = (W - P_w) // S_w + 1

    # Total length covered by the patches in height and width
    covered_height = (num_patches_h - 1) * S_h + P_h
    covered_width = (num_patches_w - 1) * S_w + P_w

    # Number of leftover pixels in height and width
    leftover_height = H - covered_height
    leftover_width = W - covered_width

    print(f'num_patches_h: {num_patches_h}, num_patches_w: {num_patches_w}, covered_h: {covered_height}, covered_w: {covered_width}, leftover_h: {leftover_height}, leftover_w: {leftover_width}\n')

    return num_patches_h, covered_height, leftover_height, num_patches_w, covered_width, leftover_width


def get_patch_coordinates(min_latitude, max_latitude, min_longitude, max_longitude, patch_size=2, stride=1):

    patch_coords = []

    current_latitude = min_latitude + 0
    while current_latitude + patch_size <= max_latitude:
        current_longitude = min_longitude + 0
        while current_longitude + patch_size <= max_longitude:
            patch_coords.append({"lat": [round(current_latitude, 2), round(current_latitude + patch_size, 2)],
                                "lon": [round(current_longitude, 2), round(current_longitude + patch_size, 2)]})
            current_longitude = current_longitude + stride
        current_latitude = current_latitude + stride

    return patch_coords


def get_lat_lon_bbox(ds):
    
    min_longitude = ds.longitude.min().values.astype(np.float64)
    max_longitude = ds.longitude.max().values.astype(np.float64)

    min_latitude = ds.latitude.min().values.astype(np.float64)
    max_latitude = ds.latitude.max().values.astype(np.float64)
    
    return min_latitude, max_latitude, min_longitude, max_longitude


def random_partition_dailytimeindex_by_ratios(days, train_part, val_part, test_part):
    # Get the total number of elements
    n = len(days)
    
    # Shuffle the indices
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Compute the split points
    split1 = np.round(n * train_part // (train_part + val_part + test_part), 0).astype(int)
    split2 = np.round(n * val_part // (train_part + val_part + test_part), 0).astype(int) + split1
    
    # Create the partitions
    part1 = indices[:split1]
    part2 = indices[split1:split2]
    part3 = indices[split2:]
    
    return part1, part2, part3


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