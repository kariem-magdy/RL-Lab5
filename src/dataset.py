import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm

class LazyRolloutDataset(Dataset):
    """
    Dataset for VAE training. 
    Loads files into memory if they fit, otherwise keeps them lazy.
    Since 1000 eps * 1000 frames * 32x32 is manageable, we try to cache.
    """
    def __init__(self, data_dir, transform=None):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # For VAE, we just need observations. 
        # Loading usually happens in the collate_fn or manually in training loop
        # to handle variable lengths.
        data = np.load(self.files[idx])
        obs = data['obs'] 
        obs_tensor = torch.from_numpy(obs).float()
        return obs_tensor, torch.zeros(1) # Dummy target

class LSTMDataset(Dataset):
    """
    Optimized Dataset for LSTM training.
    Pre-loads all data into RAM to avoid disk I/O bottleneck during training.
    """
    def __init__(self, data_dir, seq_len=100):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')])
        self.seq_len = seq_len
        self.data_cache = []
        
        print("Pre-loading LSTM Dataset into memory...")
        for f in tqdm(self.files):
            try:
                data = np.load(f)
                self.data_cache.append({
                    'obs': data['obs'],         # (T, 3, 64, 64)
                    'actions': data['actions'], # (T,)
                    'rewards': data['rewards']  # (T,)
                })
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        self.virtual_len = len(self.data_cache) * 10 # Virtual length for epoch size
        
    def __len__(self):
        return self.virtual_len
    
    def __getitem__(self, idx):
        # Randomly select an episode
        ep_idx = np.random.randint(0, len(self.data_cache))
        episode = self.data_cache[ep_idx]
        
        obs = episode['obs']
        actions = episode['actions']
        rewards = episode['rewards']
        
        total_len = len(obs)
        
        # Handle short episodes (padding or just skip)
        if total_len <= self.seq_len + 1:
            # Simple strategy: just take what we have (batching will require custom collate if variable)
            # Or enforce strict length by skipping short eps in init.
            # Here we assume most eps are long enough or we slice shorter.
            start = 0
            end = total_len
        else:
            start = np.random.randint(0, total_len - self.seq_len - 1)
            end = start + self.seq_len + 1
            
        # Slicing
        o_seq = obs[start:end]
        a_seq = actions[start:end]
        r_seq = rewards[start:end]
        
        return (torch.tensor(o_seq), 
                torch.tensor(a_seq), 
                torch.tensor(r_seq).float())