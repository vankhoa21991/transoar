"""Module containing the dataset related functionality."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from transoar.data.transforms import get_transforms


class TransoarDataset(Dataset):

    def __init__(self, data_config, split):
        assert split in ['train', 'val', 'test']
        self._data_config = data_config

        data_dir = Path("./dataset/").resolve()
        self._path_to_split = data_dir / (self._data_config['dataset_name'] + '_' + self._data_config['modality']) / split
        self._data = [data_path.name for data_path in self._path_to_split.iterdir()]

        self._augmentation = get_transforms(split)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        case = self._data[idx]
        path_to_case = self._path_to_split / case
        data_path, label_path = sorted(list(path_to_case.iterdir()), key=lambda x: len(str(x)))

        # Load npy files
        data, label = np.load(data_path), np.load(label_path)

        if self._data_config['use_augmentation']:
            data_dict = {
                'image': data,
                'label': label
            }

        # Create bboxes


        k = 12
