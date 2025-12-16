"""
Train MDN-RNN on Latent Vectors (z) and Actions.
This script must first encode all images to z using the VAE.
"""
import sys, os, yaml, torch
import numpy as np
from torch.optim import Adam
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN
from src.dataset import RolloutDataset
from torch.utils.data import DataLoader

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])
    init_wandb(config, job_type="train_rnn")

    # Load VAE
    vae = VAE(latent_dim=config['vae_latent_dim']).to(device)
    vae.load_state_dict(torch.load("vae.pth", map_location=device))
    vae.eval()

    # Load Data
    dataset = RolloutDataset('data/processed')
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Model
    action_dim = 4 # Breakout has 4 actions
    rnn = MDNRNN(latent_dim=config['vae_latent_dim'], action_dim=action_dim, hidden_dim=config['rnn_hidden_dim']).to(device)
    optimizer = Adam(rnn.parameters(), lr=config['rnn_lr'])

    print("Training RNN...")
    for epoch in range(config['rnn_epochs']):
        total_loss = 0
        for obs, actions in loader:
            obs = obs.to(device) # (B, Seq, 3, 64, 64)
            actions = actions.to(device) # (B, Seq)
            
            B, S, C, H, W = obs.shape
            
            # 1. Get Latent Z from VAE (No Grad)
            with torch.no_grad():
                obs_flat = obs.view(-1, C, H, W)
                mu, logvar = vae.encode(obs_flat)
                z = vae.reparameterize(mu, logvar)
                z = z.view(B, S, -1) # (B, Seq, Latent)

            # 2. Prepare Inputs/Targets
            # Input: z_t, a_t
            # Target: z_{t+1}
            z_in = z[:, :-1, :]
            actions_in = F.one_hot(actions[:, :-1].long(), num_classes=action_dim).float()
            z_target = z[:, 1:, :]

            # 3. Train
            optimizer.zero_grad()
            logpi, mu, sigma, _ = rnn(z_in, actions_in)
            loss = rnn.get_loss(logpi, mu, sigma, z_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            wandb.log({"rnn_loss": loss.item()})
            
        print(f"Epoch {epoch}, Loss: {total_loss}")
        torch.save(rnn.state_dict(), "rnn.pth")
    
    import torch.nn.functional as F # Re-import for safety in block context

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)