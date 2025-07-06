import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MFCCTimeseriesDataset(Dataset):
    def __init__(self, metadata_csv):
        self.data = pd.read_csv(metadata_csv)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mfcc = np.load(row['mfcc_path']).astype(np.float32)
        label = int(row['risk_level'])
        return torch.tensor(mfcc), torch.tensor(label)