import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MDNLSTM(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, hidden_dim=256, num_gaussians=5):
        super(MDNLSTM, self).__init__()
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
        
        # Reward Head: Predict next reward (Crucial for Dream training)
        self.fc_reward = nn.Linear(hidden_dim, 1)

    def forward(self, z, action, hidden=None):
        # z: (batch, seq, latent)
        # action: (batch, seq, action_dim)
        
        # Concatenate latent state and action
        x = torch.cat([z, action], dim=-1)
        
        # LSTM Forward
        output, hidden = self.lstm(x, hidden)
        
        # Project to MDN params
        logpi = self.fc_logpi(output)
        mu = self.fc_mu(output)
        logsigma = self.fc_logsigma(output)
        
        # Project to Reward
        reward_pred = self.fc_reward(output)
        
        # Reshape to (batch, seq, num_gaussians, latent_dim)
        batch, seq, _ = output.shape
        logpi = logpi.view(batch, seq, self.num_gaussians, self.latent_dim)
        mu = mu.view(batch, seq, self.num_gaussians, self.latent_dim)
        logsigma = logsigma.view(batch, seq, self.num_gaussians, self.latent_dim)
        
        logpi = F.log_softmax(logpi, dim=2) # Normalize mixing coefficients
        sigma = torch.exp(logsigma) # Ensure positive std dev
        
        return logpi, mu, sigma, reward_pred, hidden

    def get_loss(self, logpi, mu, sigma, reward_pred, target_z, target_reward):
        # target_z: (batch, seq, latent_dim)
        # target_reward: (batch, seq)
        
        # 1. MDN Loss
        target_z_exp = target_z.unsqueeze(2) # Expand for gaussians broadcasting
        
        # Log Probability: log(N(x | mu, sigma))
        log_prob = -0.5 * ((target_z_exp - mu) / sigma)**2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
        
        # Weighted Log Prob: log(pi) + log(N(x))
        weighted_log_prob = logpi + log_prob
        
        # Log Sum Exp over gaussians
        mdn_loss = -torch.logsumexp(weighted_log_prob, dim=2).mean()
        
        # 2. Reward Loss (MSE)
        reward_loss = F.mse_loss(reward_pred.squeeze(-1), target_reward)
        
        return mdn_loss + reward_loss, mdn_loss, reward_loss