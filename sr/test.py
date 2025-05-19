import sys
print('Python %s on %s' % (sys.version, sys.platform))
from tqdm import tqdm
import numpy as np
import torch
import os
import json
import argparse
import pickle
from pickle import load
from datetime import datetime
from torch.utils.data import DataLoader
from datasets import BVOCDatasetSR
from utils import set_backend, set_seed, compute_metrics, shorten_datetime
from original_san.san import SAN
from two_channel_san.san import SAN as TwoChannelSAN
from tree_channel_san.san import SAN as TreeChannelSAN
from original_hat.hat import HAT

set_backend()
set_seed(10)

BASE_OUTPUT_DIR_PATH = '<usr_dir>/runs'
BASE_DATASET_PATH = '<usr_dir>/datasets/'

########################################
# Params ArgumentParser()              #
########################################
# EXAMPLE --> <train_dataset>/x<upscaling_factor>/<model_name>/<train_run_timestamp>_<flag>
# GLOB_BIO_v1.2_sr0.5_p32_s20_zt0.95_t07-12_time/x2/SAN/20240409-155207_default_0.50

# GLOB_BIO_v3.0_sr0.25_p32_s32_zt0.95_t07-12_time/x2/SAN/default_0.25_20240409-155204
# GLOB_BIO_v3.0_sr0.25_p32_s32_zt0.95_t07-12_time/x2/SAN/default_0.25_aug_A_20240418-082304
# GLOB_BIO_v3.0_sr0.25_p32_s32_zt0.95_t07-12_time/x2/SAN/default_0.25_aug_B_20240418-082308

parser = argparse.ArgumentParser(description='BVOC Super-Resolution Testing')
# Data specs
#parser.add_argument('--qt_type', type=str, default='HR_1e3qua_fullsub', help='Type of QT to use for the dataset')
# parser.add_argument('--train_qt_flag', type=str, default='perc100_all', help='Flag for the Quantile Transformer adopted in training')
parser.add_argument('--partition_file', type=str, default='partition_index_t70_v15_te15_gs10_n200000_seed10.csv', help='Partition file for the dataset')

# Train run specs
parser.add_argument('--hr_patch_size', type=int, default=30, help='Size of high-resolution patches')
parser.add_argument('--upscaling_factor', type=int, default=2, help='Upscaling factor used in training')
parser.add_argument('--train_dataset_folder', type=str, default='p30pix_s1deg_z10_topdown_isoprene', help='Dataset used for training')
parser.add_argument('--model_name', type=str, default='SAN', help='Name of the model')
# parser.add_argument('--flag', type=str, default='default_0.25', help='Flag of the train run')
parser.add_argument('--train_run_timestamp', type=str, default='20240409-155204',
                    help='Super Resolution training run timestamp')
parser.add_argument('--run_name', type=str, default='20240828-174345_partition_index_t70_v15_te15_gs10_n200000_seed10.csv_', help='Name of the training run dir')
parser.add_argument('--channels', type=int, default=1, help='number of channels for the SAN head')

# Test specs
parser.add_argument('--batch_size', type=int, default=4000, help='Batch size')
parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')
parser.add_argument('--test_upscaling_factor', type=int, default=2, help='Upscaling factor for testing')
# parser.add_argument('--test_dataset_folder', type=str, default='p30pix_s1deg_z10_topdown_isoprene', help='Dataset that will be used for training. Usually the same as the training one')
# parser.add_argument('--unseen_areas_only', action='store_true', help='Flag to use only unseen areas in the test dataset')
parser.add_argument('--save_maps', default=True, action='store_true', help='Flag to save the output maps')
parser.add_argument('--test_type', default="", help='Type of test to perform - test, spatial or climate holdout')


args = parser.parse_args()

########################################
# Paths and Folders                    #
########################################

if args.test_type not in ['test', 'spatial', 'climate']:
    raise ValueError("Invalid test type. Choose between 'test', 'spatial' or 'climate'")

start_time = datetime.now()
run_name = args.run_name
partition_name = '_'.join(run_name.split('_')[1:-2])
fusion_flag = run_name.split('_')[-1]  # 'AG' or 'forest'
if fusion_flag not in ['AG', 'FOREST', 'ref', 'all']:
    raise ValueError("Invalid fusion flag. Should be 'AG', 'FOREST', or 'ref'. Instead got: ", fusion_flag)

model_folder = os.path.join(BASE_OUTPUT_DIR_PATH, args.train_dataset_folder, f'x{args.upscaling_factor}', args.model_name, run_name)
model_pth = os.path.join(model_folder, f'{args.model_name}_best.pth')
current_output_dir_path = os.path.join(model_folder, f'{args.test_type}_results', shorten_datetime(start_time))  # divided by dataset
map_dir_path = os.path.join(current_output_dir_path, 'maps')
os.makedirs(current_output_dir_path, exist_ok=True)
print(f'==> Output dir: {current_output_dir_path}\nRun name: {run_name} <==')

train_qt_flag = partition_name



########################################
# Variables Definition                 #
########################################
# GPU
torch.cuda.set_device(args.gpu_id)
print("Using GPU: " + str(torch.cuda.current_device()))

# Networks
network = []
if args.model_name == 'SAN':
    network_kwargs = {
        'scale': args.upscaling_factor,
        'channels': args.channels,
    }
    network = SAN(**network_kwargs)
elif args.model_name == 'HAT':
    additional_network_kwargs = {'compress_ratio': 3, # 12
                                'squeeze_factor': 30, # 12
                                'depths': (6, 6, 6, 6), # (3, 3, 3)
                                'embed_dim': 96,
                                'num_heads': (6, 6, 6, 6), # (3, 3, 3)
                                'mlp_ratio': 4., # 2
                                }
    network_kwargs = {'img_size': args.hr_patch_size // args.upscaling_factor,
                        'in_chans': args.channels,
                        'upscale': args.upscaling_factor,
                        'window_size': args.hr_patch_size // args.upscaling_factor // 3, # 5
                        'patch_size': 1, # prob 2 or 3 later
                        }
    network_kwargs.update(additional_network_kwargs)
    network = HAT(**network_kwargs)
    print('Using HAT')
else:
    print('SR Network not implemented !!!')


network.cuda()
network.load_state_dict(torch.load(model_pth, map_location=torch.device('cuda')))

# Datasets and DataLoaders
dataset_kwargs = {'root': os.path.join(BASE_DATASET_PATH, args.train_dataset_folder),
                  'partition_file': os.path.join(BASE_DATASET_PATH, args.train_dataset_folder, f'{partition_name}.csv'),
                  'quantile_transform': True,
                  'channels': args.channels,
                  'fusion_flag': fusion_flag,
                  }

dataloader_kwargs = {'batch_size': args.batch_size,
                     'num_workers': args.num_workers,
                     'pin_memory': True,
                     }

test_dataloader = DataLoader(BVOCDatasetSR(**dataset_kwargs, dataset_mode=f'{args.test_type}'), **dataloader_kwargs)

print(f'Test batches: {len(test_dataloader)}')

# Quantile Transformer
# Here a QT from LR is used. We suppose to only have LR data to be superresolved
qt_path = os.path.join(BASE_DATASET_PATH, args.train_dataset_folder, f'quantile_transformer_LR_1e3qua_fullsub_{train_qt_flag}.pkl')
qt = load(open(qt_path, 'rb'))
print(f"[QT] Loaded quantile transformer {qt_path.split('/')[-1]} for test")

########################################
# Testing                              #
########################################
start_time = datetime.now()
metrics = ['SSIM', 'PSNR', 'MSE', 'NMSE', 'MAE', 'MaxAE', 'ERGAS', 'UIQ', 'SCC', 'SRE'] # ['SSIM', 'PSNR', 'MSE', 'NMSE', 'MAE', 'MaxAE', 'ERGAS', 'UIQ', 'SCC', 'SAM', 'SRE']
metrics_complete_1 = {key: [] for key in metrics}
metrics_complete_2 = {key: [] for key in metrics}
print(f'\n\n{start_time}\tTest starts !')
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Batch', unit='batch'):
        # Extract filenames
        filelist = batch['filenames']
        filenames = filelist[0] # [0] are the HR filenames

        hr_img = batch['HR'].cuda()
        lr_img = batch['LR'].cuda()
        original_output = network(lr_img)

        # If the output has multiple channels, only use the first channel
        if original_output.shape[1] > 1:
            original_output = original_output[:, 0:1, :, :]

        # Compute metrics (mean over a single batch)
        metrics_1 = compute_metrics(original_output, hr_img, metrics)  # pre QT
        original_output = original_output.cpu().detach().numpy()
        original_hr = batch['original_HR'].cuda()
        original_output = qt.inverse_transform(original_output.reshape(-1, 1)).reshape(original_output.shape)
        original_output = torch.from_numpy(original_output).cuda()
        metrics_2 = compute_metrics(original_output, original_hr, metrics)  # post QT, aka BVOC domain

        # Save maps
        if args.save_maps:
            os.makedirs(map_dir_path, exist_ok=True)
            for i, filename in enumerate(filenames):
                hr_hat_A_i = original_output[i].cpu().numpy()
                # print(filename)
                np.save(os.path.join(map_dir_path, f'{filename}.npy'), hr_hat_A_i)

        # Append results obtained for each batch
        for key in metrics:
            metrics_complete_1[key].append(metrics_1[key])
            metrics_complete_2[key].append(metrics_2[key])

# Compute mean results of the result dictionaries
results = {}

# Flatten the list (batch) of lists (elements of the batch)
for key in metrics:
    metrics_complete_1[key] = [item for sublist in metrics_complete_1[key] for item in sublist]
    metrics_complete_2[key] = [item for sublist in metrics_complete_2[key] for item in sublist]

# Compute the mean of the metrics
for key in metrics:
    mean_1 = round(float(np.mean(metrics_complete_1[key])), 3)  # Metric 1
    mean_2 = round(float(np.mean(metrics_complete_2[key])), 3)  # Metric 2
    results[key] = [mean_1, mean_2]

# Save the results
results['metrics_1'] = metrics_complete_1
results['metrics_2'] = metrics_complete_2

# Print the mean value of all the metrics (type and position)
for key in results.keys():
    if key in metrics:
        print(f'Mean {key} {results[key]}')

# print(f'Mean results: SSIM {results["SSIM"]}, PSNR {results["PSNR"]}, MSE {results["MSE"]}, NMSE {results["NMSE"]}, MAE {results["MAE"]}')

# Save the results
with open(os.path.join(current_output_dir_path, "results.pkl"), 'wb') as file:
    pickle.dump(results, file)
# Save the arguments
with open(os.path.join(current_output_dir_path, "args.json"), 'w') as file:
    json.dump(vars(args), file)

end_time = datetime.now()
print(f'{end_time}\tTest ends !\nElapsed time: {end_time - start_time}')
print(f'==> Models in {model_folder} <==')
print(f'==> Output dir: {current_output_dir_path}\nRun name: {run_name} <==')
