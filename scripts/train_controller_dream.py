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
    controller.set_parameters(params)
    
    with torch.no_grad():
        z = torch.randn(1, 1, lstm.latent_dim).to(device)
        h = torch.zeros(1, 1, lstm.hidden_dim).to(device)
        c = torch.zeros(1, 1, lstm.hidden_dim).to(device)
        hidden = (h, c)
        
        total_reward = 0
        
        for _ in range(max_steps):
            action_idx = controller.get_action(z.squeeze(1), hidden[0].squeeze(0))
            
            action_one_hot = torch.zeros(1, 1, 4).to(device)
            action_one_hot[0, 0, action_idx] = 1
            
            logpi, mu, sigma, reward_pred, hidden = lstm(z, action_one_hot, hidden)
            
            pi = torch.exp(logpi) 
            pi_perm = pi.permute(0, 1, 3, 2) 
            cat = torch.distributions.Categorical(probs=pi_perm)
            k_indices = cat.sample() # (1, 1, L)
            
            mu_sample = torch.gather(mu, 2, k_indices.unsqueeze(2)).squeeze(2)
            sigma_sample = torch.gather(sigma, 2, k_indices.unsqueeze(2)).squeeze(2)
            
            dist = torch.distributions.Normal(mu_sample, sigma_sample)
            z_next = dist.sample() # (1, 1, L)
            
            r = reward_pred.item()
            total_reward += r
            
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
    
    controller = Controller(config['vae_latent_dim'], config['lstm_hidden_dim']).to(device)
    n_params = controller.count_parameters()
    
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.1, {'popsize': config['controller_pop_size']})
    
    best_reward = -np.inf
    print(f"Training Controller in DREAM. Generations: {config['controller_generations']}")

    for gen in range(config['controller_generations']):
        solutions = es.ask()
        rewards = []
        
        for params in solutions:
            r = dream_rollout(params, lstm, controller, device)
            rewards.append(r)
        
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