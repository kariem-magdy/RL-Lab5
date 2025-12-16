import argparse
import yaml
import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN
from src.utils.seed import set_seed
from src.utils.logging import init_wandb

class RNN_Dataset(Dataset):
    def __init__(self, data_dir, seq_len=100):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.files) * 5 # Approximate augmentation/chunking
    
    def __getitem__(self, idx):
        # Pick random file
        f_idx = idx % len(self.files)
        data = np.load(self.files[f_idx])
        obs = data['obs']
        actions = data['actions']
        
        # Pick random sequence
        if len(obs) <= self.seq_len + 1:
            start = 0
            end = len(obs)
        else:
            start = np.random.randint(0, len(obs) - self.seq_len - 1)
            end = start + self.seq_len + 1
            
        return torch.tensor(obs[start:end]), torch.tensor(actions[start:end])

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(args.seed)
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_rnn")
    
    # Load VAE (Frozen)
    vae = VAE(latent_dim=config['vae_latent_dim']).to(device)
    vae.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "vae_latest.pth"), map_location=device))
    vae.eval()
    
    dataset = RNN_Dataset(config['data_processed'], seq_len=config['sequence_len'])
    loader = DataLoader(dataset, batch_size=config['rnn_batch_size'], num_workers=2)
    
    rnn = MDNRNN(
        latent_dim=config['vae_latent_dim'], 
        action_dim=4, # Breakout
        hidden_dim=config['rnn_hidden_dim'],
        num_gaussians=config['rnn_num_gaussians']
    ).to(device)
    
    optimizer = Adam(rnn.parameters(), lr=config['rnn_lr'])
    
    print("Starting RNN Training...")
    for epoch in range(config['rnn_epochs']):
        rnn.train()
        total_loss = 0
        
        for batch_idx, (obs, actions) in enumerate(loader):
            obs = obs.to(device)
            actions = actions.to(device)
            
            # 1. Encode images to Z using VAE (No gradient)
            with torch.no_grad():
                B, Seq, C, H, W = obs.shape
                obs_flat = obs.view(-1, C, H, W)
                mu, logvar = vae.encode(obs_flat)
                z = vae.reparameterize(mu, logvar)
                z = z.view(B, Seq, -1)
                
            # 2. Prepare Inputs/Targets
            # Input: z_t, a_t
            # Target: z_{t+1}
            z_in = z[:, :-1, :] # 0 to T-1
            actions_in = F.one_hot(actions[:, :-1].long(), num_classes=4).float()
            z_target = z[:, 1:, :] # 1 to T
            
            # 3. Train
            optimizer.zero_grad()
            logpi, mu, sigma, _ = rnn(z_in, actions_in)
            loss = rnn.get_loss(logpi, mu, sigma, z_target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if args.log and batch_idx % 10 == 0:
                wandb.log({"rnn/loss": loss.item()})
                
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader):.4f}")
        torch.save(rnn.state_dict(), os.path.join(config['checkpoint_dir'], "rnn_latest.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)