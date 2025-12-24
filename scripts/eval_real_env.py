import argparse
import yaml
import torch
import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
import cv2
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_lstm import MDNLSTM
from src.models.controller import Controller
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb 

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = torch.device(config['device'])
    if args.log: init_wandb(config, job_type="eval")

    vae_path = os.path.join(config['checkpoint_dir'], "vae_best.pth")
    lstm_path = os.path.join(config['checkpoint_dir'], "lstm_best.pth")
    
    if not os.path.exists(vae_path) or not os.path.exists(lstm_path):
        print("❌ Models not found.")
        return

    vae = VAE(latent_dim=config['vae_latent_dim'], resize_dim=config['resize_dim']).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'])
    vae.eval()

    lstm = MDNLSTM(config['vae_latent_dim'], 4, config['lstm_hidden_dim'], config['lstm_num_gaussians']).to(device)
    lstm.load_state_dict(torch.load(lstm_path, map_location=device)['model_state_dict'])
    lstm.eval()

    controller = Controller(config['vae_latent_dim'], config['lstm_hidden_dim']).to(device)
    if not os.path.exists(args.checkpoint):
        print(f"❌ Controller checkpoint not found: {args.checkpoint}")
        return
    
    controller.load_state_dict(torch.load(args.checkpoint, map_location=device))
    controller.eval()

    env = gym.make(config['env_name'], render_mode='rgb_array')
    results = []

    print(f"Starting Evaluation for {args.episodes} episodes...")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        h = torch.zeros(1, 1, config['lstm_hidden_dim']).to(device)
        c = torch.zeros(1, 1, config['lstm_hidden_dim']).to(device)
        hidden = (h, c)

        while not done:
            frame = cv2.resize(obs, (64, 64))
            frame = torch.tensor(frame).float().permute(2, 0, 1) / 255.0
            frame = frame.unsqueeze(0).to(device)

            with torch.no_grad():
                mu, _ = vae.encode(frame)
                z = mu 
                action_idx = controller.get_action(z, hidden[0].squeeze(0))
                
                obs, reward, terminated, truncated, _ = env.step(action_idx)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                action_one_hot = torch.zeros(1, 1, 4).to(device)
                action_one_hot[0, 0, action_idx] = 1
                _, _, _, _, hidden = lstm(z.unsqueeze(0), action_one_hot, hidden)

        print(f"Episode {episode+1}: Reward {total_reward}, Duration {steps}")
        results.append({"episode": episode, "episode_reward": total_reward, "duration": steps})
        
        if args.log:
            wandb.log({
                "test/episode_reward": total_reward,
                "test/episode_duration": steps,
                "test/episode": episode
            })

    df = pd.DataFrame(results)
    os.makedirs(config['eval_dir'], exist_ok=True)
    df.to_csv(os.path.join(config['eval_dir'], "eval_results.csv"), index=False)
    print("✅ Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)