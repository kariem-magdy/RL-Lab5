import argparse
import yaml
import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE, vae_loss_function
from src.dataset import LazyRolloutDataset
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb

def train_epoch(model, loader, optimizer, device, config, epoch):
    model.train()
    total_loss = 0
    total_frames = 0
    
    for batch_idx, (obs, _) in enumerate(loader):
        obs = obs.squeeze(0).to(device) # (Seq, 3, 64, 64)
        
        chunk_size = config['vae_batch_size']
        for i in range(0, len(obs), chunk_size):
            batch = obs[i:i+chunk_size]
            if len(batch) < 2: continue

            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch)
            loss, mse, kld = vae_loss_function(recon_x, batch, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)
            total_frames += len(batch)
            
    return total_loss / total_frames if total_frames > 0 else 0

def validate(model, loader, device, config):
    model.eval()
    total_loss = 0
    total_frames = 0
    with torch.no_grad():
        for (obs, _) in loader:
            obs = obs.squeeze(0).to(device)
            chunk_size = config['vae_batch_size']
            for i in range(0, len(obs), chunk_size):
                batch = obs[i:i+chunk_size]
                recon_x, mu, logvar = model(batch)
                loss, _, _ = vae_loss_function(recon_x, batch, mu, logvar)
                total_loss += loss.item() * len(batch)
                total_frames += len(batch)
    return total_loss / total_frames if total_frames > 0 else 0

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    set_seed(args.seed)
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_vae")

    # Data
    full_dataset = LazyRolloutDataset(config['data_processed'])
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    # Model & Optimization
    vae = VAE(latent_dim=config['vae_latent_dim']).to(device)
    optimizer = Adam(vae.parameters(), lr=config['vae_lr'])
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0
    
    # Resumption
    ckpt_path = os.path.join(config['checkpoint_dir'], "vae_latest.pth")
    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device)
        vae.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    print("Starting VAE Training...")
    for epoch in range(start_epoch, config['vae_epochs']):
        train_loss = train_epoch(vae, train_loader, optimizer, device, config, epoch)
        val_loss = validate(vae, val_loader, device, config)
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
        if args.log:
            wandb.log({"vae/train_loss": train_loss, "vae/val_loss": val_loss})

        # Save Latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        
        # Save Best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(vae.state_dict(), os.path.join(config['checkpoint_dir'], "vae_best.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)