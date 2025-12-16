"""
LSTM + Mixture Density Network (MDN) implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, hidden_dim=256, num_gaussians=5):
        super(MDNRNN, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        
        # MDN Heads
        self.fc_logpi = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

    def forward(self, z, action, hidden=None):
        # z: (batch, seq, latent_dim)
        # action: (batch, seq, action_dim) - One Hot
        
        # Combine state and action
        x = torch.cat([z, action], dim=-1)
        output, hidden = self.lstm(x, hidden)
        
        # Predict next latent state params
        logpi = self.fc_logpi(output)
        mu = self.fc_mu(output)
        logsigma = self.fc_logsigma(output)
        
        # Reshape to (batch, seq, num_gaussians, latent_dim)
        batch_size, seq_len, _ = output.shape
        logpi = logpi.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        mu = mu.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        logsigma = logsigma.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        
        logpi = F.log_softmax(logpi, dim=2) # Normalize weights
        sigma = torch.exp(logsigma)
        
        return logpi, mu, sigma, hidden

    def get_loss(self, logpi, mu, sigma, target_z):
        # target_z: (batch, seq, latent_dim)
        target_z = target_z.unsqueeze(2) # Expand for gaussians
        
        # Log Probability of target under each gaussian
        # (x - mu)^2 / (2 * sigma^2)
        log_prob = -0.5 * ((target_z - mu) / sigma)**2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
        
        # Combine with weights (log-sum-exp trick)
        # log( sum( pi * N(x) ) ) = log_sum_exp( log_pi + log_N(x) )
        loss = torch.logsumexp(logpi + log_prob, dim=2)
        return -torch.mean(loss) # Negative Log Likelihood