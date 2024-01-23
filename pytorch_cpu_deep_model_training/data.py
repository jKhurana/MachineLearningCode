from torch.utils.data.dataset import Dataset
import numpy as np
import re
import string

import os
import pickle
from data_util import *


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.n_data = self.data.shape[0]

    def __len__(self):
        return self.n_data

    def __getitem__(self, index: int):
        if index < 0 or index>=self.n_data:
            index = index % self.n_data
        
        return self.data.iloc[index]