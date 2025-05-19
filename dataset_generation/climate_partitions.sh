#!/bin/bash

# for all partitions
TRAIN=75000
VAL=5000
SAMPLES=100000
CLIMATE=10000
SPATIAL=10000
SEED=15
DATASET_PATH="<usr_dir>/<dataset_name>"

# Med values for arguments
PARTITION_TAG="med"
EXCLUDE_LIST="6 7 8 9"
INCLUDE_LIST="0 4 5 6 7 8 9 18"
HOLDOUT_POINT="15 50"

# Construct the command to call the Python script with the parsed arguments
COMMAND="python3 partition_generation_climate.py \
--train_samples ${TRAIN} \
--val_samples ${VAL} \
--samples ${SAMPLES} \
--climate_cv_size ${CLIMATE} \
--spatial_cv_size ${SPATIAL} \
--partition_tag ${PARTITION_TAG} \
--exclude_list ${EXCLUDE_LIST} \
--include_list ${INCLUDE_LIST} \
--seed ${SEED} \
--dataset_path ${DATASET_PATH} \
--holdout_point ${HOLDOUT_POINT}"

# Print the command for debugging
echo "Running command: ${COMMAND}"

# Run the Python script with the constructed command
eval $COMMAND

# russia values for arguments
PARTITION_TAG="26"
EXCLUDE_LIST="26"
INCLUDE_LIST="26"

# Construct the command to call the Python script with the parsed arguments
COMMAND="python3 partition_generation_climate.py \
--train_samples ${TRAIN} \
--val_samples ${VAL} \
--samples ${SAMPLES} \
--climate_cv_size ${CLIMATE} \
--spatial_cv_size ${SPATIAL} \
--partition_tag ${PARTITION_TAG} \
--exclude_list ${EXCLUDE_LIST} \
--include_list ${INCLUDE_LIST} \
--seed ${SEED} \
--dataset_path ${DATASET_PATH}"

# Print the command for debugging
echo "Running command: ${COMMAND}"

# Run the Python script with the constructed command
eval $COMMAND

# channel values for arguments
PARTITION_TAG="15"
EXCLUDE_LIST="15"
INCLUDE_LIST="0 15"

# Construct the command to call the Python script with the parsed arguments
COMMAND="python3 partition_generation_climate.py \
--train_samples ${TRAIN} \
--val_samples ${VAL} \
--samples ${SAMPLES} \
--climate_cv_size ${CLIMATE} \
--spatial_cv_size ${SPATIAL} \
--partition_tag ${PARTITION_TAG} \
--exclude_list ${EXCLUDE_LIST} \
--include_list ${INCLUDE_LIST} \
--seed ${SEED} \
--dataset_path ${DATASET_PATH}"

# Print the command for debugging
echo "Running command: ${COMMAND}"

# Run the Python script with the constructed command
eval $COMMAND

# channel values for arguments
PARTITION_TAG="27"
EXCLUDE_LIST="27"
INCLUDE_LIST="0 27"
HOLDOUT_POINT="0 40"

# Construct the command to call the Python script with the parsed arguments
COMMAND="python3 partition_generation_climate.py \
--train_samples ${TRAIN} \
--val_samples ${VAL} \
--samples ${SAMPLES} \
--climate_cv_size ${CLIMATE} \
--spatial_cv_size ${SPATIAL} \
--partition_tag ${PARTITION_TAG} \
--exclude_list ${EXCLUDE_LIST} \
--include_list ${INCLUDE_LIST} \
--seed ${SEED} \
--dataset_path ${DATASET_PATH} \
--holdout_point ${HOLDOUT_POINT}"

# Print the command for debugging
echo "Running command: ${COMMAND}"

# Run the Python script with the constructed command
eval $COMMAND
