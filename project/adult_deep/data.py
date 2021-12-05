from enum import auto, Enum
from os.path import join
import numpy as np
import pandas as pd
import pickle
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class DatasetChoice(Enum):
    ADULT = auto()
    HOUSING = auto()


def get_housing_dataloader(data_path, rows_fname, labels_fname, batch_size):
    rows = pd.read_pickle(join(data_path, rows_fname))
    rows = torch.from_numpy(rows.astype(np.float32).values)
    labels = pd.read_pickle(join(data_path, labels_fname))
    labels = torch.from_numpy(labels.astype(np.float32).values).unsqueeze(dim=1)
    dataset = TensorDataset(rows, labels)
    return DataLoader(dataset, batch_size=batch_size)


def get_adult_data_loader(rows, labels, batch_size):
    rows = Tensor(rows)
    labels = Tensor(labels)
    dataset = TensorDataset(rows, labels)
    return DataLoader(dataset, batch_size=batch_size)


def load_housing_data(data_path, batch_size=128):
    train_dataloader = get_housing_dataloader(data_path, 'X_test.gz', 'y_test.gz', batch_size)
    cross_val_dataloader = get_housing_dataloader(data_path, 'X_test.gz', 'y_test.gz', batch_size)
    test_dataloader = get_housing_dataloader(data_path, 'X_test.gz', 'y_test.gz', batch_size)

    return train_dataloader, cross_val_dataloader, test_dataloader


def load_adult_data(data_path, batch_size=128):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    train_dataloader = get_adult_data_loader(data['train_rows'], data['train_labels'], batch_size)

    cross_val_dataloader = get_adult_data_loader(data['cross_val_rows'], data['cross_val_labels'], batch_size)

    test_dataloader = get_adult_data_loader(data['test_rows'], data['test_labels'], batch_size)

    return train_dataloader, cross_val_dataloader, test_dataloader


def load_data(dataset_choice: DatasetChoice, data_path, batch_size):
    if dataset_choice == DatasetChoice.ADULT:
        return load_adult_data(data_path, batch_size=batch_size)
    elif dataset_choice == DatasetChoice.HOUSING:
        return load_housing_data(data_path, batch_size=batch_size)
    else:
        raise NotImplementedError('Dataset not supported.')
