import torch
from torch.utils.data import Dataset
import numpy as np
import os

class LazyRolloutDataset(Dataset):
    """
    Dataset for VAE training. 
    Loads files on-demand using memory mapping to prevent OOM.
    """
    def __init__(self, data_dir, transform=None):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load in mmap mode to avoid reading entire file to RAM
        try:
            data = np.load(self.files[idx], mmap_mode='r')
            obs = data['obs'] 
            # Convert to tensor (forces a read of only the specific array)
            obs_tensor = torch.from_numpy(np.array(obs)).float()
            return obs_tensor, torch.zeros(1) 
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(1, 3, 64, 64), torch.zeros(1)

class LSTMDataset(Dataset):
    """
    Memory-Efficient Dataset for LSTM training.
    Uses lazy loading and random slicing to handle large datasets.
    """
    def __init__(self, data_dir, seq_len=100, virtual_size=10000):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.seq_len = seq_len
        self.virtual_size = virtual_size
        
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
            
    def __len__(self):
        return self.virtual_size
    
    def __getitem__(self, idx):
        # Randomly select a file (infinite sampling strategy)
        file_idx = np.random.randint(0, len(self.files))
        filepath = self.files[file_idx]
        
        try:
            data = np.load(filepath, mmap_mode='r')
            obs = data['obs']         # (T, 3, 64, 64)
            actions = data['actions'] # (T,)
            rewards = data['rewards'] # (T,)
            
            total_len = len(obs)
            
            if total_len <= self.seq_len + 1:
                # Retry if episode is too short
                return self.__getitem__((idx + 1) % self.virtual_size)
            
            # Random window
            start = np.random.randint(0, total_len - self.seq_len - 1)
            end = start + self.seq_len + 1
            
            # Read only the slice from disk
            o_seq = np.array(obs[start:end])
            a_seq = np.array(actions[start:end])
            r_seq = np.array(rewards[start:end])
            
            return (torch.tensor(o_seq).float(), 
                    torch.tensor(a_seq).float(), 
                    torch.tensor(r_seq).float())
                    
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return self.__getitem__((idx + 1) % self.virtual_size)