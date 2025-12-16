import argparse
import yaml
import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE, vae_loss_function
from src.dataset import LazyRolloutDataset
from src.utils.seed import set_seed
from src.utils.logging import init_wandb

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    set_seed(args.seed)
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_vae")

    # Data - Use Lazy Loading
    dataset = LazyRolloutDataset(config['data_processed'])
    
    # Collate function to handle variable sequence lengths if we want to batch episodes
    # Ideally for VAE we just want a bag of frames. 
    # Current LazyRolloutDataset returns (Seq, 3, 64, 64).
    # Since sequences differ in length, standard DataLoader batching fails unless batch_size=1.
    # We will use batch_size=1 (one episode per batch) and then flatten internally.
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Model
    vae = VAE(latent_dim=config['vae_latent_dim']).to(device)
    optimizer = Adam(vae.parameters(), lr=config['vae_lr'])
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    print("Starting VAE Training...")
    for epoch in range(config['vae_epochs']):
        vae.train()
        total_loss = 0
        total_frames = 0
        
        for batch_idx, (obs, _) in enumerate(loader):
            # obs shape: (1, Seq, 3, 64, 64)
            obs = obs.squeeze(0).to(device) # (Seq, 3, 64, 64)
            
            # Sub-batching to avoid GPU OOM on long episodes
            chunk_size = config['vae_batch_size']
            
            for i in range(0, len(obs), chunk_size):
                batch = obs[i:i+chunk_size]
                
                optimizer.zero_grad()
                recon_x, mu, logvar = vae(batch)
                loss, mse, kld = vae_loss_function(recon_x, batch, mu, logvar)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_frames += len(batch)
            
            if args.log and batch_idx % 10 == 0:
                wandb.log({"vae/loss": loss.item() / len(batch)})

        avg_loss = total_loss / total_frames
        print(f"Epoch {epoch+1}: Avg Loss per Frame {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(vae.state_dict(), os.path.join(config['checkpoint_dir'], "vae_latest.pth"))
        
        # Log Visuals
        if args.log:
            with torch.no_grad():
                sample = obs[:8]
                recon, _, _ = vae(sample)
                comparison = torch.cat([sample, recon], dim=0)
                wandb.log({"reconstruction": [wandb.Image(comparison, caption="Top: Orig, Bot: Recon")]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)