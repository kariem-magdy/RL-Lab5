import argparse
import gymnasium as gym
import ale_py
import numpy as np
import os
import sys
import yaml
import time
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.seed import set_seed

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    
    # Ensure raw data dir exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    env = gym.make(config['env_name'])
    
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    print(f"Collecting {args.episodes} episodes...")
    
    for i in tqdm(range(args.episodes)):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_dones = []
        
        steps = 0
        while not (done or truncated) and steps < config['max_frames']:
            # Random policy for World Model training data
            action = env.action_space.sample()
            
            # Store data
            ep_obs.append(obs) # Store uint8 raw frame to save space until preprocess
            ep_actions.append(action)
            
            obs, reward, done, truncated, _ = env.step(action)
            ep_rewards.append(reward)
            ep_dones.append(done or truncated)
            steps += 1
            
        all_obs.append(np.array(ep_obs))
        all_actions.append(np.array(ep_actions))
        all_rewards.append(np.array(ep_rewards))
        all_dones.append(np.array(ep_dones))

    # Save as object array because lengths differ
    np.savez_compressed(
        args.out,
        obs=np.array(all_obs, dtype=object),
        actions=np.array(all_actions, dtype=object),
        rewards=np.array(all_rewards, dtype=object),
        dones=np.array(all_dones, dtype=object)
    )
    print(f"Saved raw rollouts to {args.out}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="data/raw/rollouts.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)