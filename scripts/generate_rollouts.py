import argparse
import gymnasium as gym
import ale_py
import numpy as np
import os
import sys
import yaml
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.seed import set_seed

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Note: We rely on the env's internal seeding or seed manually below
    env = gym.make(config['env_name'])
    
    # Set seeds
    set_seed(args.seed)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    
    # Create raw dir
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Collecting {args.episodes} episodes to {args.out_dir}...")
    
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
            action = env.action_space.sample()
            
            # Store raw obs (uint8) to save space
            ep_obs.append(obs)
            ep_actions.append(action)
            
            obs, reward, done, truncated, _ = env.step(action)
            ep_rewards.append(reward)
            ep_dones.append(done or truncated)
            steps += 1
            
        # Save IMMEDIATELY per episode to avoid OOM
        save_path = os.path.join(args.out_dir, f"rollout_{i}.npz")
        np.savez_compressed(
            save_path,
            obs=np.array(ep_obs), # uint8
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
    args = parser.parse_args()
    main(args)