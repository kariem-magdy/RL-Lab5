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
from src.models.mdn_rnn import MDNRNN
from src.models.controller import Controller
from scripts.preprocess import preprocess_frame
from src.utils.seed import set_seed
from src.utils.logging import init_wandb

# Worker function to evaluate a single candidate
def evaluate_candidate(params, env, vae, rnn, controller, device):
    controller.set_parameters(params)
    
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    # Initial Hidden State
    h = torch.zeros(1, 1, rnn.hidden_dim).to(device)
    c = torch.zeros(1, 1, rnn.hidden_dim).to(device)
    hidden = (h, c)
    
    steps = 0
    while not (done or truncated) and steps < 1000:
        # Preprocess
        frame = preprocess_frame(obs)
        frame_tensor = torch.tensor(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(frame_tensor)
            z = mu # Use mean for deterministic policy
            
            # Get Action from Controller
            action_idx = controller.get_action(z, hidden[0].squeeze(0))
            
            # Step Env
            obs, reward, done, truncated, _ = env.step(action_idx)
            total_reward += reward
            
            # Step RNN to update hidden state
            action_one_hot = torch.zeros(1, 1, 4).to(device)
            action_one_hot[0, 0, action_idx] = 1
            _, _, _, hidden = rnn(z.unsqueeze(0), action_one_hot, hidden)
        
        steps += 1
        
    return total_reward

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(args.seed)
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="train_controller")
    
    # Load Models
    vae = VAE(config['vae_latent_dim']).to(device)
    vae.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "vae_latest.pth"), map_location=device))
    
    rnn = MDNRNN(config['vae_latent_dim'], 4, config['rnn_hidden_dim']).to(device)
    rnn.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "rnn_latest.pth"), map_location=device))
    
    controller = Controller(config['vae_latent_dim'], config['rnn_hidden_dim']).to(device)
    
    # Environment
    env = gym.make(config['env_name'])
    
    # CMA-ES
    n_params = controller.count_parameters()
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.1, {
        'popsize': config['controller_pop_size']
    })
    
    print(f"Training Controller with CMA-ES (Pop: {config['controller_pop_size']})...")
    
    best_reward = -np.inf
    
    for gen in range(config['controller_generations']):
        solutions = es.ask()
        rewards = []
        
        # Parallelize this in production (using multiprocessing)
        # Here sequential for simplicity/portability
        for params in solutions:
            r = evaluate_candidate(params, env, vae, rnn, controller, device)
            rewards.append(r)
            
        # CMA minimizes
        es.tell(solutions, [-r for r in rewards])
        
        avg_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        
        print(f"Gen {gen}: Avg Reward {avg_reward:.2f}, Max {max_reward:.2f}")
        if args.log:
            wandb.log({"controller/avg_reward": avg_reward, "controller/max_reward": max_reward})
            
        # Save best
        if max_reward > best_reward:
            best_reward = max_reward
            controller.set_parameters(es.result.xbest)
            torch.save(controller.state_dict(), os.path.join(config['checkpoint_dir'], "controller_best.pth"))
            
        if best_reward > config['target_return']:
            print("Target return reached.")
            break
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)