"""
Train Controller using CMA-ES.
"""
import sys, os, yaml, torch
import gymnasium as gym
import cma
import numpy as np
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN
from src.models.controller import Controller
from src.utils.misc import preprocess_frame

def evaluate(env, vae, rnn, controller, device):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    # Initial Hidden State
    h = torch.zeros(1, 1, rnn.hidden_dim).to(device) # (NumLayers, Batch, Hidden)
    c = torch.zeros(1, 1, rnn.hidden_dim).to(device)
    hidden = (h, c)
    
    while not done:
        # 1. Process Frame -> VAE
        frame = preprocess_frame(obs)
        frame = torch.tensor(frame).unsqueeze(0).to(device) # (1, 3, 64, 64)
        with torch.no_grad():
            mu, logvar = vae.encode(frame)
            z = vae.reparameterize(mu, logvar) # (1, Latent)
            
        # 2. Controller Action
        # Current hidden state is from previous step (h_{t-1})
        # We need h_t for controller, which comes from RNN(z_t, a_{t-1}, h_{t-1})
        # Simplified: World Models usually feeds z_t and h_{t-1} to Controller
        action_idx = controller.get_action(z, hidden[0].squeeze(0))
        
        # 3. Step Env
        obs, reward, done, _, _ = env.step(action_idx)
        total_reward += reward
        
        # 4. Update RNN Hidden State
        action_one_hot = torch.zeros(1, 1, 4).to(device)
        action_one_hot[0, 0, action_idx] = 1
        
        with torch.no_grad():
            _, _, _, hidden = rnn(z.unsqueeze(0), action_one_hot, hidden)
            
    return total_reward

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])
    init_wandb(config, job_type="train_controller")
    
    # Load Models
    vae = VAE(latent_dim=config['vae_latent_dim']).to(device)
    vae.load_state_dict(torch.load("vae.pth", map_location=device))
    
    rnn = MDNRNN(latent_dim=config['vae_latent_dim'], action_dim=4, hidden_dim=config['rnn_hidden_dim']).to(device)
    rnn.load_state_dict(torch.load("rnn.pth", map_location=device))
    
    controller = Controller(latent_dim=config['vae_latent_dim'], hidden_dim=config['rnn_hidden_dim']).to(device)
    
    # CMA-ES Setup
    n_params = controller.count_parameters()
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.1, {'popsize': config['pop_size']})
    
    env = gym.make(config['env_name'])
    
    print(f"Starting CMA-ES for {config['n_generations']} generations...")
    
    for gen in range(config['n_generations']):
        solutions = es.ask()
        rewards = []
        
        for params in solutions:
            controller.set_parameters(params)
            # Evaluate 1 episode (average of 3 is better but slower)
            r = evaluate(env, vae, rnn, controller, device)
            rewards.append(r)
            
        # CMA-ES minimizes, so negate rewards
        es.tell(solutions, [-r for r in rewards])
        
        avg_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        
        print(f"Gen {gen}: Avg: {avg_reward}, Max: {max_reward}")
        wandb.log({"avg_reward": avg_reward, "max_reward": max_reward})
        
        # Save best
        controller.set_parameters(es.result.xbest)
        torch.save(controller.state_dict(), "controller.pth")
        
        if max_reward > config['target_return']:
            print("Target return reached!")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)