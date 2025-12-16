"""
Generates random rollouts for VAE and RNN training.
"""
import argparse
import gymnasium as gym
import ale_py
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.misc import set_seed, preprocess_frame

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    
    env = gym.make(config['env_name'])
    out_dir = 'data/processed'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {config['rollout_samples']} rollouts...")
    
    for i in tqdm(range(config['rollout_samples'])):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        obs_seq = []
        action_seq = []
        
        while not (done or truncated):
            # Random policy for data collection
            action = env.action_space.sample()
            
            # Process Frame
            processed_obs = preprocess_frame(obs)
            obs_seq.append(processed_obs)
            action_seq.append(action)
            
            obs, reward, done, truncated, _ = env.step(action)
            
        # Save compressed
        np.savez_compressed(
            os.path.join(out_dir, f'rollout_{i}.npz'),
            obs=np.array(obs_seq),
            actions=np.array(action_seq)
        )
    
    env.close()
    print("Rollouts generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)