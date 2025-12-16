import torch
from torch.utils.data import Dataset
import numpy as np
import os

class LazyRolloutDataset(Dataset):
    """
    Lazy loads .npz files from disk to prevent OOM.
    Used for VAE training.
    """
    def __init__(self, data_dir, transform=None):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        obs = data['obs'] 
        actions = data['actions']
        
        obs_tensor = torch.from_numpy(obs).float()
        actions_tensor = torch.from_numpy(actions).float()
        
        return obs_tensor, actions_tensor

class LSTMDataset(Dataset):
    """
    Dataset for LSTM training.
    Samples random sequences.
    """
    def __init__(self, data_dir, seq_len=100):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.seq_len = seq_len
        self.virtual_len = len(self.files) * 5 
        
    def __len__(self):
        return self.virtual_len
    
    def __getitem__(self, idx):
        file_idx = np.random.randint(0, len(self.files))
        data = np.load(self.files[file_idx])
        
        obs = data['obs']
        actions = data['actions']
        
        total_len = len(obs)
        if total_len <= self.seq_len + 1:
            start = 0
            end = total_len
        else:
            start = np.random.randint(0, total_len - self.seq_len - 1)
            end = start + self.seq_len + 1
            
        return torch.tensor(obs[start:end]), torch.tensor(actions[start:end])