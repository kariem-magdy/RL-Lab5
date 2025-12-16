import torch
import torch.nn as nn
import numpy as np

class Controller(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=4):
        super(Controller, self).__init__()
        # Simple linear policy mapping [z, h] -> action logits
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        # z: (1, latent_dim)
        # h: (1, hidden_dim)
        inp = torch.cat([z, h], dim=1)
        return self.fc(inp)

    def get_action(self, z, h):
        with torch.no_grad():
            logits = self.forward(z, h)
            # Deterministic for now, can add temperature
            action = torch.argmax(logits, dim=1).item()
        return action
    
    def set_parameters(self, flatten_params):
        """Load flattened parameters from CMA-ES optimizer."""
        state_dict = self.state_dict()
        idx = 0
        for name, param in state_dict.items():
            count = param.numel()
            # Convert numpy param to tensor
            new_param = torch.from_numpy(flatten_params[idx:idx+count]).float()
            state_dict[name].copy_(new_param.view(param.shape))
            idx += count
            
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())