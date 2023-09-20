"""Script to prepare amos dataset."""

import logging

from pathlib import Path
import random

from transoar.utils.io import get_config, set_root_logger, load_json
from transoar.data.preprocessor_luna16 import PreProcessor


if __name__ == "__main__":
    # Set config of root logger
    set_root_logger('./logs/prepare_dataset.log')
    logging.info('Started preparing dataset.')
    
    # Load data config
    preprocessing_config = get_config('preprocessing_luna16')
    data_config = get_config(preprocessing_config['dataset_config'])

    print(preprocessing_config)
    print(data_config)

    random.seed(preprocessing_config['seed'])  # Set arbitrary seed to make experiments reproducible

    dataset_name = preprocessing_config['dataset_name']
    fold = preprocessing_config['fold']
    path_dataset = Path(preprocessing_config['path_to_dataset'])   # complete dataset 
    path_to_splits = Path(f"{preprocessing_config['path_to_split']}/{dataset_name}_{fold}")
    split_file = Path(f"{preprocessing_config['path_to_dataset']}/preprocessed/splits_final.json")
    logging.info(f'path_dataset: {path_dataset}')
    logging.info(f'path_to_splits: {path_to_splits}')
    logging.info(f'fold: {fold}')
    logging.info(f'dataset_name: {dataset_name}')


    split_info = load_json(split_file)[fold]
    
    # Create test, val, and train split
    train_set = split_info['train']
    val_set = split_info['val']
    test_set = split_info['val']

    logging.info(f'Preparing dataset {dataset_name}_{fold}.')
    logging.info(f'len train: {len(train_set)}, len val: {len(val_set)}, len test: {len(test_set)}.')

    # Prepare dataset based on dataset analysis
    logging.info(f"Starting dataset preprocessing. Target shape: {preprocessing_config['resize_shape']}.")
    preprocessor = PreProcessor(
        train_set, val_set, test_set, path_dataset, path_to_splits, preprocessing_config, data_config
    )
    preprocessor.run()
    logging.info(f'Succesfully finished dataset preprocessing.')
