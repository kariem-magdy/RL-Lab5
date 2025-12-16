import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.mdn_lstm import MDNLSTM

def test_lstm_shapes():
    lstm = MDNLSTM(latent_dim=32, action_dim=4, hidden_dim=256, num_gaussians=5)
    z = torch.randn(1, 10, 32) # Batch 1, Seq 10
    a = torch.randn(1, 10, 4)
    
    logpi, mu, sigma, hidden = lstm(z, a)
    
    # Output should be (Batch, Seq, Gaussians, Latent)
    assert logpi.shape == (1, 10, 5, 32)
    assert mu.shape == (1, 10, 5, 32)
    assert sigma.shape == (1, 10, 5, 32)
    
    # Check hidden state (h, c)
    assert isinstance(hidden, tuple)
    assert hidden[0].shape == (1, 1, 256)
    assert hidden[1].shape == (1, 1, 256)