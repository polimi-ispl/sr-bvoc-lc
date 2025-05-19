import os
import numpy as np
import pandas as pd
from datetime import datetime

# Partition generation for dataset without holdout sets

partition_logic = {'train': 75, 'val': 5, 'test': 20, 'samples': 100000} # percentages TODO make into argparse

partition_mode = 'random' # random | sequential
seed = 15

print("loading dataset index...")
dataset_path = '<usr_dir>/<dataset_name>'

dataset_index = pd.read_csv(os.path.join(dataset_path, 'dataset_index.csv'), index_col=0)
dataset_partition_index = dataset_index.copy()

rng = np.random.default_rng(seed=seed)

sample_size = partition_logic['samples']
random_sample = dataset_partition_index.sample(n=sample_size, random_state=rng)

# Partition the dataset into train, validation, and test sets
print("Partitioning the dataset...")
train_percentage = partition_logic['train']
val_percentage = partition_logic['val']
test_percentage = partition_logic['test']

train_size = int(sample_size * train_percentage / 100)
val_size = int(sample_size * val_percentage / 100)
test_size = int(sample_size * test_percentage / 100)

random_index = random_sample.index.tolist()
rng.shuffle(random_index)
train_index = random_index[:train_size]
val_index = random_index[train_size:train_size + val_size]
test_index = random_index[train_size + val_size:train_size + val_size + test_size]

dataset_partition_index['partition'] = None
dataset_partition_index.loc[train_index, 'partition'] = 'train'
dataset_partition_index.loc[val_index, 'partition'] = 'val'
dataset_partition_index.loc[test_index, 'partition'] = 'test'

print("train samples:", dataset_partition_index[dataset_partition_index['partition'] == 'train'].shape[0])
print("val samples:", dataset_partition_index[dataset_partition_index['partition'] == 'val'].shape[0])
print("test samples:", dataset_partition_index[dataset_partition_index['partition'] == 'test'].shape[0])

# Save the partitioned dataset index
current_time = datetime.now().strftime("%Y%m%d_%H%M")
filename = f'fullgeo_partition_index_t{train_percentage}_v{val_percentage}_te{test_percentage}_n{sample_size}_seed{seed}_{current_time}.csv'
print(f"Saving partitioned dataset index to {filename}")
dataset_partition_index.to_csv(os.path.join(dataset_path, filename))
