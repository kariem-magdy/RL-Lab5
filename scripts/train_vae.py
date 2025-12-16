"""
Train the VAE on generated rollouts.
"""
import sys, os, yaml, torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE, loss_function
from src.dataset import RolloutDataset
from src.utils.tracking import init_wandb

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['device'])
    init_wandb(config, job_type="train_vae")

    # Data
    dataset = RolloutDataset('data/processed')
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    # Model
    vae = VAE(latent_dim=config['vae_latent_dim']).to(device)
    optimizer = Adam(vae.parameters(), lr=config['vae_lr'])

    print("Training VAE...")
    for epoch in range(config['vae_epochs']):
        vae.train()
        total_loss = 0
        for batch_idx, (obs, _) in enumerate(loader):
            # Flatten batch and sequence: (B, Seq, C, H, W) -> (B*Seq, C, H, W)
            obs = obs.to(device)
            B, S, C, H, W = obs.shape
            obs = obs.view(-1, C, H, W)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(obs)
            loss = loss_function(recon_x, obs, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                wandb.log({"vae_loss": loss.item() / (B*S)})
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")
        
        # Save Checkpoint
        torch.save(vae.state_dict(), "vae.pth")
        
        # Log reconstruction
        with torch.no_grad():
             wandb.log({"reconstruction": [wandb.Image(recon_x[0], caption="Recon"), wandb.Image(obs[0], caption="Original")]})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)