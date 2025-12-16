import argparse
import yaml
import torch
import os
import sys
import numpy as np
import cma
import gymnasium as gym
import wandb
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_lstm import MDNLSTM
from src.models.controller import Controller
from scripts.preprocess import preprocess_frame
from src.utils.seed import set_seed
from src.utils.logging import init_wandb

def evaluate_candidate(params, env, vae, lstm, controller, device):
    controller.set_parameters(params)
    
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    h = torch.zeros(1, 1, lstm.hidden_dim).to(device)
    c = torch.zeros(1, 1, lstm.hidden_dim).to(device)
    hidden = (h, c)
    
    steps = 0
    while not (done or truncated) and steps < 1000:
        frame = preprocess_frame(obs)
        frame_tensor = torch.tensor(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(frame_tensor)
            z = mu 
            
            action_idx = controller.get_action(z, hidden[0].squeeze(0))
            
            obs, reward, done, truncated, _ = env.step(action_idx)
            total_reward += reward
            
            action_one_hot = torch.zeros(1, 1, 4).to(device)
            action_one_hot[0, 0, action_idx] = 1
            _, _, _, hidden = lstm(z.unsqueeze(0), action_one_hot, hidden)
        
        steps += 1
        
    return total_reward

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(args.seed)
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_controller")
    
    vae = VAE(config['vae_latent_dim']).to(device)
    vae.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "vae_latest.pth"), map_location=device))
    
    lstm = MDNLSTM(config['vae_latent_dim'], 4, config['lstm_hidden_dim']).to(device)
    lstm.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "lstm_latest.pth"), map_location=device))
    
    controller = Controller(config['vae_latent_dim'], config['lstm_hidden_dim']).to(device)
    
    env = gym.make(config['env_name'])
    
    n_params = controller.count_parameters()
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.1, {
        'popsize': config['controller_pop_size']
    })
    
    print(f"Training Controller (Pop: {config['controller_pop_size']})...")
    
    best_reward = -np.inf
    
    for gen in range(config['controller_generations']):
        solutions = es.ask()
        rewards = []
        
        for params in solutions:
            r = evaluate_candidate(params, env, vae, lstm, controller, device)
            rewards.append(r)
            
        es.tell(solutions, [-r for r in rewards])
        
        avg_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        
        print(f"Gen {gen}: Avg {avg_reward:.2f}, Max {max_reward:.2f}")
        if args.log:
            wandb.log({"controller/avg_reward": avg_reward, "controller/max_reward": max_reward})
            
        if max_reward > best_reward:
            best_reward = max_reward
            controller.set_parameters(es.result.xbest)
            torch.save(controller.state_dict(), os.path.join(config['checkpoint_dir'], "controller_best.pth"))
            
        if best_reward > config['target_return']:
            print("Target reached.")
            break
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)