import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.mdn_rnn import MDNRNN

def test_rnn_shapes():
    rnn = MDNRNN(latent_dim=32, action_dim=4, hidden_dim=256, num_gaussians=5)
    z = torch.randn(1, 10, 32) # Batch 1, Seq 10
    a = torch.randn(1, 10, 4)
    
    logpi, mu, sigma, hidden = rnn(z, a)
    
    # Output should be (Batch, Seq, Gaussians, Latent)
    assert logpi.shape == (1, 10, 5, 32)
    assert mu.shape == (1, 10, 5, 32)
    assert sigma.shape == (1, 10, 5, 32)