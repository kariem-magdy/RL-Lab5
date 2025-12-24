import argparse
import yaml
import torch
import os
import sys
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_lstm import MDNLSTM
from src.dataset import LSTMDataset
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb

def train_epoch(model, loader, optimizer, vae, device):
    model.train()
    total_loss = 0
    total_mdn = 0
    total_reward = 0
    
    for batch_idx, (obs, actions, rewards) in enumerate(loader):
        obs = obs.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        
        with torch.no_grad():
            B, Seq, C, H, W = obs.shape
            obs_flat = obs.view(-1, C, H, W)
            mu, _ = vae.encode(obs_flat)
            z = mu.view(B, Seq, -1)
            
        z_in = z[:, :-1, :]
        actions_in = F.one_hot(actions[:, :-1].long(), num_classes=4).float()
        
        z_target = z[:, 1:, :]
        rewards_target = rewards[:, 1:]
        
        optimizer.zero_grad()
        logpi, mu_mdn, sigma_mdn, reward_pred, _ = model(z_in, actions_in)
        
        loss, mdn_loss, reward_loss = model.get_loss(logpi, mu_mdn, sigma_mdn, reward_pred, z_target, rewards_target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping
        optimizer.step()
        
        total_loss += loss.item()
        total_mdn += mdn_loss.item()
        total_reward += reward_loss.item()
        
    return total_loss / len(loader), total_reward / len(loader)

def validate(model, loader, vae, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (obs, actions, rewards) in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            
            B, Seq, C, H, W = obs.shape
            mu, _ = vae.encode(obs.view(-1, C, H, W))
            z = mu.view(B, Seq, -1)
            
            z_in = z[:, :-1, :]
            actions_in = F.one_hot(actions[:, :-1].long(), num_classes=4).float()
            
            z_target = z[:, 1:, :]
            rewards_target = rewards[:, 1:]
            
            logpi, mu_mdn, sigma_mdn, reward_pred, _ = model(z_in, actions_in)
            loss, _, _ = model.get_loss(logpi, mu_mdn, sigma_mdn, reward_pred, z_target, rewards_target)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_lstm")
    
    vae = VAE(latent_dim=config['vae_latent_dim'], resize_dim=config.get('resize_dim', 64)).to(device)
    vae_path = os.path.join(config['checkpoint_dir'], "vae_best.pth")
    if not os.path.exists(vae_path):
        print(f"Error: VAE checkpoint not found at {vae_path}")
        sys.exit(1)
        
    vae_ckpt = torch.load(vae_path, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    
    full_dataset = LSTMDataset(config['data_processed'], seq_len=config['sequence_len'])
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=config['lstm_batch_size'], num_workers=0) 
    val_loader = DataLoader(val_set, batch_size=config['lstm_batch_size'], num_workers=0)
    
    lstm = MDNLSTM(
        latent_dim=config['vae_latent_dim'], 
        action_dim=4, 
        hidden_dim=config['lstm_hidden_dim'],
        num_gaussians=config['lstm_num_gaussians']
    ).to(device)
    
    optimizer = Adam(lstm.parameters(), lr=config['lstm_lr'])
    
    ckpt_path = os.path.join(config['checkpoint_dir'], "lstm_latest.pth")
    best_path = os.path.join(config['checkpoint_dir'], "lstm_best.pth")
    best_loss = float('inf')
    start_epoch = 0
    patience = 5
    patience_counter = 0
    
    if os.path.exists(ckpt_path):
        print("Resuming LSTM...")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            lstm.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
        except Exception as e:
            print(f"Error loading LSTM checkpoint: {e}. Restarting.")

    print("Starting LSTM Training...")
    for epoch in range(start_epoch, config['lstm_epochs']):
        train_loss, reward_loss = train_epoch(lstm, train_loader, optimizer, vae, device)
        val_loss = validate(lstm, val_loader, vae, device)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if args.log:
            wandb.log({"lstm/loss": train_loss, "lstm/val_loss": val_loss, "lstm/reward_mse": reward_loss})
            
        save_dict = {
            'epoch': epoch,
            'model_state_dict': lstm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }
        torch.save(save_dict, ckpt_path)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(save_dict, best_path)
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