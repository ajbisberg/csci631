import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from preprocess_data import load_data


class AdultDataset(Dataset):
    def __init__(self, fname):
        self.rows, self.labels = load_data(fname)
        self.rows = torch.from_numpy(self.rows.astype(np.float32))
        self.labels = torch.from_numpy(self.labels.astype(np.float32))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index], self.labels[index]
