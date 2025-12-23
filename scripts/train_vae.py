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

def train_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss = 0
    total_frames = 0
    
    for batch_idx, (obs, _) in enumerate(loader):
        # obs is (Batch=1, Seq, C, H, W)
        obs = obs.squeeze(0).to(device)
        
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

def save_checkpoint(model, optimizer, epoch, config, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, path)

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_vae")

    # Data
    full_dataset = LazyRolloutDataset(config['data_processed'])
    if len(full_dataset) == 0:
        print("No data found. Exiting.")
        sys.exit(1)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    # Model
    vae = VAE(latent_dim=config['vae_latent_dim'], resize_dim=config.get('resize_dim', 64)).to(device)
    optimizer = Adam(vae.parameters(), lr=config['vae_lr'])
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    ckpt_path = os.path.join(config['checkpoint_dir'], "vae_latest.pth")
    best_path = os.path.join(config['checkpoint_dir'], "vae_best.pth")
    
    start_epoch = 0
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint...")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if 'config' in ckpt and ckpt['config']['vae_latent_dim'] != config['vae_latent_dim']:
                raise ValueError("Config mismatch in checkpoint")
            vae.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")

    print("Starting VAE Training...")
    for epoch in range(start_epoch, config['vae_epochs']):
        train_loss = train_epoch(vae, train_loader, optimizer, device, config, epoch)
        val_loss = validate(vae, val_loader, device, config)
        
        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        
        if args.log:
            wandb.log({"vae/train_loss": train_loss, "vae/val_loss": val_loss})

        save_checkpoint(vae, optimizer, epoch, config, ckpt_path)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(vae, optimizer, epoch, config, best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)