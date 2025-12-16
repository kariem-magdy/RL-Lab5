import argparse
import yaml
import torch
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE, vae_loss_function
from src.utils.seed import set_seed
from src.utils.logging import init_wandb

class VAE_Dataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        # Load all into memory for VAE training (images are small enough if N=1000)
        # Alternatively, lazy load.
        print("Loading dataset into memory...")
        self.data = []
        for f in self.files:
            dat = np.load(f)['obs'] # (Seq, 3, 64, 64)
            self.data.append(dat)
        self.data = np.concatenate(self.data, axis=0) # (TotalFrames, 3, 64, 64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    set_seed(args.seed)
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_vae")

    # Data
    dataset = VAE_Dataset(config['data_processed'])
    loader = DataLoader(dataset, batch_size=config['vae_batch_size'], shuffle=True, num_workers=2)

    # Model
    vae = VAE(latent_dim=config['vae_latent_dim']).to(device)
    optimizer = Adam(vae.parameters(), lr=config['vae_lr'])
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    print("Starting VAE Training...")
    for epoch in range(config['vae_epochs']):
        vae.train()
        total_loss = 0
        total_mse = 0
        total_kld = 0
        
        for batch_idx, obs in enumerate(loader):
            obs = obs.to(device)
            optimizer.zero_grad()
            
            recon_x, mu, logvar = vae(obs)
            loss, mse, kld = vae_loss_function(recon_x, obs, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()

            if args.log and batch_idx % 50 == 0:
                wandb.log({
                    "vae/loss": loss.item() / len(obs),
                    "vae/mse": mse.item() / len(obs),
                    "vae/kld": kld.item() / len(obs)
                })

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(vae.state_dict(), os.path.join(config['checkpoint_dir'], "vae_latest.pth"))
        
        # Log Visuals
        if args.log:
            with torch.no_grad():
                # Get first 8 images
                orig = obs[:8]
                recon = recon_x[:8]
                comparison = torch.cat([orig, recon], dim=0)
                wandb.log({"reconstruction": [wandb.Image(comparison, caption="Top: Orig, Bot: Recon")]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)