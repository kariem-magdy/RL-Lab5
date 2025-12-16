import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE

def test_vae_shapes():
    vae = VAE(latent_dim=32)
    x = torch.randn(2, 3, 64, 64) # Batch 2
    recon, mu, logvar = vae(x)
    
    assert recon.shape == x.shape
    assert mu.shape == (2, 32)
    assert logvar.shape == (2, 32)