import sys
sys.path.append('<usr_dir>')
import glob
import random
import torch
import os
import numpy as np
import pandas as pd
from pickle import load
from torch.utils.data import Dataset
from dataset_generation.utils import compute_quantile_transformation

class BVOCDatasetSR(Dataset):
    def __init__(self, root,
                 partition_file,
                 dataset_mode='train',
                 randomized_order = True,
                 seed=42,
                 quantile_transform=False,
                 channels=1,
                 fusion_flag='AG',
                 qt_folder=None,
                 qt_type=None,
                 train_qt_flag=None):
        
        self.dataset_mode = dataset_mode  # train | val | test | spatial | climate
        self.quantile_transform = quantile_transform
        self.randomized_order = randomized_order
        self.seed = seed
        self.channels = channels
        self.fusion_flag = fusion_flag
        # self.qt_folder = qt_folder
        # self.qt_type = qt_type

        print(f"\nDataset mode: {self.dataset_mode}")

        partition_index = pd.read_csv(partition_file, index_col=0)

        # ———————— File lists ————————
        if self.dataset_mode == 'spatial':
            if self.randomized_order:
                partition_index = partition_index[partition_index['geosplit'] == 'spatial']
                partition_index = partition_index.sample(n=10000, random_state=self.seed)
            
            hr_filepaths = sorted(partition_index[partition_index['geosplit'] == 'spatial']['HR_file_path'].values)
            lr_filepaths = sorted(partition_index[partition_index['geosplit'] == 'spatial']['LR_file_path'].values)
            ag_filepaths = sorted(partition_index[partition_index['geosplit'] == 'spatial']['AG_file_path'].values)
            forest_filepaths = sorted(partition_index[partition_index['geosplit'] == 'spatial']['FOREST_file_path'].values)
        elif self.dataset_mode == 'climate':
            if self.randomized_order:
                partition_index = partition_index[partition_index['geosplit'] == 'climate']
                partition_index = partition_index.sample(n=10000, random_state=self.seed)
            
            hr_filepaths = sorted(partition_index[partition_index['geosplit'] == 'climate']['HR_file_path'].values)
            lr_filepaths = sorted(partition_index[partition_index['geosplit'] == 'climate']['LR_file_path'].values)
            ag_filepaths = sorted(partition_index[partition_index['geosplit'] == 'climate']['AG_file_path'].values)
            forest_filepaths = sorted(partition_index[partition_index['geosplit'] == 'climate']['FOREST_file_path'].values)
        elif self.dataset_mode == 'test':
            if self.randomized_order:
                partition_index = partition_index[partition_index['partition'] == 'test']
                partition_index = partition_index.sample(n=10000, random_state=self.seed)
            
            hr_filepaths = sorted(partition_index[partition_index['partition'] == 'test']['HR_file_path'].values)
            lr_filepaths = sorted(partition_index[partition_index['partition'] == 'test']['LR_file_path'].values)
            ag_filepaths = sorted(partition_index[partition_index['partition'] == 'test']['AG_file_path'].values)
            forest_filepaths = sorted(partition_index[partition_index['partition'] == 'test']['FOREST_file_path'].values)
        else:
            if self.randomized_order:
                partition_index = partition_index.sample(frac=1, random_state=self.seed)
                
            hr_filepaths = sorted(partition_index[partition_index['partition'] == self.dataset_mode]['HR_file_path'].values)
            lr_filepaths = sorted(partition_index[partition_index['partition'] == self.dataset_mode]['LR_file_path'].values)
            ag_filepaths = sorted(partition_index[partition_index['partition'] == self.dataset_mode]['AG_file_path'].values)
            forest_filepaths = sorted(partition_index[partition_index['partition'] == self.dataset_mode]['FOREST_file_path'].values)

        self.files = list(zip(hr_filepaths, lr_filepaths, ag_filepaths, forest_filepaths))

        qt_flag_name = os.path.splitext(os.path.basename(partition_file))[0] # to match runs

        # ———————— Quantile transformation ————————
        if self.quantile_transform and self.dataset_mode == 'train':
            # Compute quantile transformation from scratch and save it
            self.qt, _ = compute_quantile_transformation(root, self.files, flag_name=qt_flag_name)  # use the HR one
            print(f"[QT] Computed quantile transformer for training — from HR maps")
        elif self.quantile_transform and self.dataset_mode == 'val':
            # or load the precomputed quantile transformers, the one used for training
            qt_path = os.path.join(root, f'quantile_transformer_HR_1e3qua_fullsub_{qt_flag_name}.pkl')
            self.qt = load(open(qt_path, 'rb'))
            print(f"[QT] Loaded quantile transformer {qt_path.split('/')[-1]} for validation")
        elif self.quantile_transform and self.dataset_mode == 'test':
            # load a user defined quantile transformer for the test set
            qt_path = os.path.join(root, f'quantile_transformer_LR_1e3qua_fullsub_{qt_flag_name}.pkl')
            self.qt = load(open(qt_path, 'rb'))
            print(f"[QT] Loaded quantile transformer {qt_path.split('/')[-1]} for test")
        elif self.quantile_transform and self.dataset_mode == 'spatial':
            # load a user defined quantile transformer for the test set
            qt_path = os.path.join(root, f'quantile_transformer_LR_1e3qua_fullsub_{qt_flag_name}.pkl')
            self.qt = load(open(qt_path, 'rb'))
            print(f"[QT] Loaded quantile transformer {qt_path.split('/')[-1]} for test")
        elif self.quantile_transform and self.dataset_mode == 'climate':
            # load a user defined quantile transformer for the test set
            qt_path = os.path.join(root, f'quantile_transformer_LR_1e3qua_fullsub_{qt_flag_name}.pkl')
            self.qt = load(open(qt_path, 'rb'))
            print(f"[QT] Loaded quantile transformer {qt_path.split('/')[-1]} for test")

    def __getitem__(self, index):
        # Paths
        hr_path, lr_path, ag_path, forest_path = self.files[index]
        # Filename
        filenames = (os.path.basename(hr_path).split('.')[0], os.path.basename(lr_path).split('.')[0], os.path.basename(ag_path).split('.')[0])
        # Load
        hr_img = np.load(hr_path)
        lr_img = np.load(lr_path)
        ag_img = np.load(ag_path)
        forest_img = np.load(forest_path)
        # Expand dims (1-channel)
        hr_img = np.expand_dims(hr_img, axis=0)
        lr_img = np.expand_dims(lr_img, axis=0)
        ag_img = np.expand_dims(ag_img, axis=0)
        forest_img = np.expand_dims(forest_img, axis=0)

        lr_img[lr_img < 0.0] = 0.0  # negative emissions are meaningless
        ag_img[ag_img < 0.0] = 0.0  # negative agricultural areas are meaningless
        forest_img[forest_img < 0.0] = 0.0  # negative forest areas are meaningless

        # Quantile transform
        if self.quantile_transform:
            hr_img_n = self.qt.transform(hr_img.reshape(-1, 1)).reshape(hr_img.shape)
            lr_img_n = self.qt.transform(lr_img.reshape(-1, 1)).reshape(lr_img.shape)
        else:  # No quantile transform
            hr_img_n = hr_img
            lr_img_n = lr_img

        if self.channels == 2:
            if self.fusion_flag == 'AG':
                # stack isoprene and ag to the 2-channel LR image
                lr_img_n = np.concatenate((lr_img_n, ag_img), axis=0)
            elif self.fusion_flag == 'FOREST':
                # stack isoprene and forest to the 2-channel LR image
                lr_img_n = np.concatenate((lr_img_n, forest_img), axis=0)
        
        if self.channels == 3:
            lr_img_n = np.concatenate((lr_img_n, ag_img, forest_img), axis=0)

        data = {'HR': hr_img_n.astype(np.float32),
                'LR': lr_img_n.astype(np.float32),
                'filenames': filenames}
        if self.dataset_mode in ['test', 'spatial', 'climate']:
            if self.channels == 1:
                data['original_HR'] = hr_img.astype(np.float32)
                data['original_LR'] = lr_img.astype(np.float32)   
            elif self.channels == 2:
                data['original_HR'] = hr_img.astype(np.float32)
                data['original_LR'] = lr_img.astype(np.float32)
                if self.fusion_flag == 'AG':
                    data['original_AG'] = ag_img.astype(np.float32)
                elif self.fusion_flag == 'FOREST':
                    data['original_FOREST'] = forest_img.astype(np.float32)
            elif self.channels == 3:
                data['original_HR'] = hr_img.astype(np.float32)
                data['original_LR'] = lr_img.astype(np.float32)
                data['original_AG'] = ag_img.astype(np.float32)
                data['original_FOREST'] = forest_img.astype(np.float32)   

        return data

    def __len__(self):
        return len(self.files)
