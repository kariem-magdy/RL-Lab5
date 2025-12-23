import argparse
import gymnasium as gym
import ale_py
import numpy as np
import os
import sys
import yaml
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.seed import set_seed
from src.models.vae import VAE
from src.models.mdn_lstm import MDNLSTM
from src.models.controller import Controller
from src.utils.image import preprocess_frame

def get_action(config, obs, vae, lstm, controller, hidden, device, random_policy=True):
    if random_policy:
        return np.random.randint(0, 4), hidden
    
    # Model-based policy
    with torch.no_grad():
        frame = preprocess_frame(obs)
        frame_tensor = torch.tensor(frame).unsqueeze(0).to(device)
        mu, _ = vae.encode(frame_tensor)
        z = mu
        
        # Controller action
        action_idx = controller.get_action(z, hidden[0].squeeze(0))
        
        # Update LSTM state
        action_one_hot = torch.zeros(1, 1, 4).to(device)
        action_one_hot[0, 0, action_idx] = 1
        _, _, _, _, hidden = lstm(z.unsqueeze(0), action_one_hot, hidden)
        
    return action_idx, hidden

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = gym.make(config['env_name'])
    set_seed(args.seed)
    
    # Policy Setup
    random_policy = True
    vae, lstm, controller = None, None, None
    device = torch.device(config['device'])
    
    if args.policy:
        print(f"Loading policy from {args.policy} for iterative data collection...")
        random_policy = False
        
        vae = VAE(config['vae_latent_dim']).to(device)
        vae.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "vae_best.pth"), map_location=device))
        
        lstm = MDNLSTM(config['vae_latent_dim'], 4, config['lstm_hidden_dim']).to(device)
        lstm.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "lstm_best.pth"), map_location=device))
        
        controller = Controller(config['vae_latent_dim'], config['lstm_hidden_dim']).to(device)
        controller.load_state_dict(torch.load(args.policy, map_location=device))
        
        vae.eval()
        lstm.eval()
        
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Collecting {args.episodes} episodes...")
    
    for i in tqdm(range(args.episodes)):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        
        # Init LSTM state
        if not random_policy:
            h = torch.zeros(1, 1, config['lstm_hidden_dim']).to(device)
            c = torch.zeros(1, 1, config['lstm_hidden_dim']).to(device)
            hidden = (h, c)
        else:
            hidden = None
        
        steps = 0
        while not (done or truncated) and steps < config['max_frames']:
            action, hidden = get_action(config, obs, vae, lstm, controller, hidden, device, random_policy)
            
            ep_obs.append(obs) # Store raw
            ep_actions.append(action)
            
            obs, reward, done, truncated, _ = env.step(action)
            ep_rewards.append(reward)
            ep_dones.append(done or truncated)
            steps += 1
            
        save_path = os.path.join(args.out_dir, f"rollout_{i}.npz")
        np.savez_compressed(
            save_path,
            obs=np.array(ep_obs),
            actions=np.array(ep_actions),
            rewards=np.array(ep_rewards),
            dones=np.array(ep_dones)
        )

    env.close()
    print("Rollout collection complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", help="Path to controller checkpoint for iterative training", default=None)
    args = parser.parse_args()
    main(args)