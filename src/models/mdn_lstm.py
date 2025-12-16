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

        # Input: z (latent) + action
        self.lstm = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        
        # MDN Heads: Output params for Gaussian Mixture
        self.fc_logpi = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_logsigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

    def forward(self, z, action, hidden=None):
        # z: (batch, seq, latent)
        # action: (batch, seq, action_dim)
        # Note: action should be one-hot encoded before passing here
        
        x = torch.cat([z, action], dim=-1)
        output, hidden = self.lstm(x, hidden)
        
        # Project to MDN params
        logpi = self.fc_logpi(output)
        mu = self.fc_mu(output)
        logsigma = self.fc_logsigma(output)
        
        # Reshape to (batch, seq, num_gaussians, latent_dim)
        batch, seq, _ = output.shape
        logpi = logpi.view(batch, seq, self.num_gaussians, self.latent_dim)
        mu = mu.view(batch, seq, self.num_gaussians, self.latent_dim)
        logsigma = logsigma.view(batch, seq, self.num_gaussians, self.latent_dim)
        
        logpi = F.log_softmax(logpi, dim=2) # Normalize mixing coefficients
        sigma = torch.exp(logsigma) # Ensure positive std dev
        
        return logpi, mu, sigma, hidden

    def get_loss(self, logpi, mu, sigma, target_z):
        # target_z: (batch, seq, latent_dim)
        target_z = target_z.unsqueeze(2) # Expand for gaussians broadcasting
        
        # Log Probability: log(N(x | mu, sigma))
        # Constant term -0.5*log(2pi) can be ignored for optimization but included for correctness
        log_prob = -0.5 * ((target_z - mu) / sigma)**2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
        
        # Weighted Log Prob: log(pi) + log(N(x))
        weighted_log_prob = logpi + log_prob
        
        # Log Sum Exp over gaussians to get log likelihood of mixture
        # Sum over latent dimensions (usually assume independent gaussians per dimension)
        loss = torch.logsumexp(weighted_log_prob, dim=2)
        
        # Negative Log Likelihood
        return -torch.mean(loss)