import argparse
import yaml
import torch
import os
import sys
import numpy as np
import cma
import torch.nn.functional as F
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.mdn_lstm import MDNLSTM
from src.models.controller import Controller
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb

def dream_rollout(params, lstm, controller, device, max_steps=1000):
    """
    Simulates an episode entirely in the latent space (Dream).
    Returns total accumulated predicted reward.
    """
    controller.set_parameters(params)
    
    with torch.no_grad():
        # Initialize Latent State (Random z or zeros)
        # z shape: (1, 1, latent_dim)
        z = torch.randn(1, 1, lstm.latent_dim).to(device)
        
        # Initialize Hidden State
        h = torch.zeros(1, 1, lstm.hidden_dim).to(device)
        c = torch.zeros(1, 1, lstm.hidden_dim).to(device)
        hidden = (h, c)
        
        total_reward = 0
        
        for _ in range(max_steps):
            # 1. Controller chooses action based on current z and h
            # controller expects z: (1, latent) and h: (1, hidden)
            # hidden[0] is (1, 1, hidden_dim) -> squeeze to (1, hidden_dim)
            action_idx = controller.get_action(z.squeeze(1), hidden[0].squeeze(0))
            
            # 2. Prepare inputs for LSTM
            action_one_hot = torch.zeros(1, 1, 4).to(device)
            action_one_hot[0, 0, action_idx] = 1
            
            # 3. Step LSTM to predict next z and reward
            logpi, mu, sigma, reward_pred, hidden = lstm(z, action_one_hot, hidden)
            
            # 4. Sample next z from Mixture of Gaussians
            # logpi: (1, 1, K, L)
            pi = torch.exp(logpi) 
            
            # Sample gaussian index k for each latent dim l
            # Permute to (1, 1, L, K) for categorical
            pi_perm = pi.permute(0, 1, 3, 2) 
            cat = torch.distributions.Categorical(probs=pi_perm)
            k_indices = cat.sample() # (1, 1, L)
            
            # Gather mu and sigma
            # mu: (1, 1, K, L) -> gather along dim 2
            mu_sample = torch.gather(mu, 2, k_indices.unsqueeze(2)).squeeze(2)
            sigma_sample = torch.gather(sigma, 2, k_indices.unsqueeze(2)).squeeze(2)
            
            dist = torch.distributions.Normal(mu_sample, sigma_sample)
            z_next = dist.sample() # (1, 1, L)
            
            # 5. Accumulate Reward
            r = reward_pred.item()
            total_reward += r
            
            # 6. Stability Check / Termination
            if torch.isnan(z_next).any() or torch.abs(z_next).max() > 100:
                break
                
            z = z_next
            
        return total_reward

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_controller_dream")
    
    # Load LSTM (The World Model)
    lstm_path = os.path.join(config['checkpoint_dir'], "lstm_best.pth")
    if not os.path.exists(lstm_path):
        print(f"Error: LSTM checkpoint not found at {lstm_path}. Train LSTM first.")
        sys.exit(1)

    lstm = MDNLSTM(
        config['vae_latent_dim'], 
        4, 
        config['lstm_hidden_dim'], 
        config['lstm_num_gaussians']
    ).to(device)
    
    ckpt = torch.load(lstm_path, map_location=device)
    lstm.load_state_dict(ckpt['model_state_dict'])
    lstm.eval() 
    
    # Init Controller
    controller = Controller(config['vae_latent_dim'], config['lstm_hidden_dim']).to(device)
    
    n_params = controller.count_parameters()
    print(f"Training Controller in DREAM. Params: {n_params}")
    
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.1, {'popsize': config['controller_pop_size']})
    
    best_reward = -np.inf
    
    for gen in range(config['controller_generations']):
        solutions = es.ask()
        rewards = []
        
        # This loop can be parallelized, but sequential for safety in basic implementation
        for params in solutions:
            r = dream_rollout(params, lstm, controller, device)
            rewards.append(r)
        
        # CMA-ES minimizes, so we pass negative rewards
        es.tell(solutions, [-r for r in rewards])
        
        avg_r = np.mean(rewards)
        max_r = np.max(rewards)
        
        print(f"Gen {gen}: Avg Dream Reward {avg_r:.2f}, Max {max_r:.2f}")
        if args.log:
            wandb.log({"dream/avg_reward": avg_r, "dream/max_reward": max_r})
            
        if max_r > best_reward:
            best_reward = max_r
            controller.set_parameters(es.result.xbest)
            torch.save(controller.state_dict(), os.path.join(config['checkpoint_dir'], "controller_dream_best.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)