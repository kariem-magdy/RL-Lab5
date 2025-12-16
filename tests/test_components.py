import pytest
import torch
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN
from src.models.controller import Controller
from src.utils.misc import preprocess_frame
import numpy as np

def test_preprocess():
    dummy_frame = np.zeros((210, 160, 3), dtype=np.uint8)
    processed = preprocess_frame(dummy_frame)
    assert processed.shape == (3, 64, 64)
    assert processed.dtype == np.float32
    assert processed.max() <= 1.0

def test_vae_forward():
    vae = VAE(latent_dim=32)
    dummy_in = torch.randn(2, 3, 64, 64)
    recon, mu, logvar = vae(dummy_in)
    assert recon.shape == dummy_in.shape
    assert mu.shape == (2, 32)

def test_rnn_forward():
    rnn = MDNRNN(latent_dim=32, action_dim=4, hidden_dim=256)
    z = torch.randn(1, 10, 32) # Batch 1, Seq 10
    a = torch.randn(1, 10, 4)
    logpi, mu, sigma, hidden = rnn(z, a)
    assert mu.shape == (1, 10, 5, 32) # 5 gaussians
    assert hidden[0].shape == (1, 1, 256) # 1 layer, batch 1, hidden

def test_controller():
    ctrl = Controller(latent_dim=32, hidden_dim=256, action_dim=4)
    z = torch.randn(1, 32)
    h = torch.randn(1, 256)
    action = ctrl.get_action(z, h)
    assert isinstance(action, int)
    assert 0 <= action < 4