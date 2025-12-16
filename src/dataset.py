import torch
from torch.utils.data import Dataset
import numpy as np
import os

class RolloutDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        obs = data['obs'] # (Seq, 3, 64, 64)
        actions = data['actions'] # (Seq,)
        return torch.tensor(obs), torch.tensor(actions)