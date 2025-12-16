import argparse
import yaml
import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_lstm import MDNLSTM
from src.dataset import LSTMDataset
from src.utils.seed import set_seed
from src.utils.logging import init_wandb

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(args.seed)
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_lstm")
    
    # Load VAE (Frozen)
    vae = VAE(latent_dim=config['vae_latent_dim']).to(device)
    vae.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "vae_latest.pth"), map_location=device))
    vae.eval()
    
    # Dataset
    dataset = LSTMDataset(config['data_processed'], seq_len=config['sequence_len'])
    loader = DataLoader(dataset, batch_size=config['lstm_batch_size'], num_workers=2)
    
    # Model
    lstm = MDNLSTM(
        latent_dim=config['vae_latent_dim'], 
        action_dim=4, 
        hidden_dim=config['lstm_hidden_dim'],
        num_gaussians=config['lstm_num_gaussians']
    ).to(device)
    
    optimizer = Adam(lstm.parameters(), lr=config['lstm_lr'])
    
    print("Starting LSTM Training...")
    for epoch in range(config['lstm_epochs']):
        lstm.train()
        total_loss = 0
        
        for batch_idx, (obs, actions) in enumerate(loader):
            obs = obs.to(device)
            actions = actions.to(device)
            
            with torch.no_grad():
                B, Seq, C, H, W = obs.shape
                obs_flat = obs.view(-1, C, H, W)
                mu, _ = vae.encode(obs_flat)
                z = mu 
                z = z.view(B, Seq, -1)
                
            z_in = z[:, :-1, :]
            actions_in = F.one_hot(actions[:, :-1].long(), num_classes=4).float()
            z_target = z[:, 1:, :]
            
            optimizer.zero_grad()
            logpi, mu_mdn, sigma_mdn, _ = lstm(z_in, actions_in)
            loss = lstm.get_loss(logpi, mu_mdn, sigma_mdn, z_target)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if args.log and batch_idx % 10 == 0:
                wandb.log({"lstm/loss": loss.item()})
                
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader):.4f}")
        torch.save(lstm.state_dict(), os.path.join(config['checkpoint_dir'], "lstm_latest.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)