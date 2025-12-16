import argparse
import yaml
import torch
import os
import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import pandas as pd
import time
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN
from src.models.controller import Controller
from scripts.preprocess import preprocess_frame
from src.utils.logging import init_wandb

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])
    
    if args.log: init_wandb(config, job_type="eval")
    
    # Load Models
    vae = VAE(config['vae_latent_dim']).to(device)
    vae.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "vae_latest.pth"), map_location=device))
    
    rnn = MDNRNN(config['vae_latent_dim'], 4, config['rnn_hidden_dim']).to(device)
    rnn.load_state_dict(torch.load(os.path.join(config['checkpoint_dir'], "rnn_latest.pth"), map_location=device))
    
    controller = Controller(config['vae_latent_dim'], config['rnn_hidden_dim']).to(device)
    controller.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Setup Env with Recording
    os.makedirs(config['video_dir'], exist_ok=True)
    env = gym.make(config['env_name'], render_mode='rgb_array')
    env = RecordVideo(env, video_folder=config['video_dir'], episode_trigger=lambda x: True, name_prefix="eval_run")
    
    results = []
    
    print(f"Evaluating for {args.episodes} episodes...")
    for i in range(args.episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        start_time = time.time()
        total_reward = 0
        steps = 0
        
        # RNN Hidden init
        h = torch.zeros(1, 1, config['rnn_hidden_dim']).to(device)
        c = torch.zeros(1, 1, config['rnn_hidden_dim']).to(device)
        hidden = (h, c)
        
        while not (done or truncated):
            frame = preprocess_frame(obs)
            frame_tensor = torch.tensor(frame).unsqueeze(0).to(device)
            
            with torch.no_grad():
                mu, _ = vae.encode(frame_tensor)
                action = controller.get_action(mu, hidden[0].squeeze(0))
                
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                
                action_one_hot = torch.zeros(1, 1, 4).to(device)
                action_one_hot[0, 0, action] = 1
                _, _, _, hidden = rnn(mu.unsqueeze(0), action_one_hot, hidden)
            steps += 1
            
        duration = time.time() - start_time
        results.append({
            "episode_reward": total_reward,
            "episode_length": steps,
            "episode_duration_seconds": duration
        })
        print(f"Ep {i}: Reward {total_reward}, Duration {duration:.2f}s")
        
    env.close()
    
    # Save CSV
    df = pd.DataFrame(results)
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, f"eval_{int(time.time())}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    if args.log:
        wandb.log({"eval_table": wandb.Table(dataframe=df)})
        # Upload videos
        video_files = [f for f in os.listdir(config['video_dir']) if f.endswith(".mp4")]
        for v in video_files:
            wandb.log({"eval_video": wandb.Video(os.path.join(config['video_dir'], v))})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", default="results/eval_runs/")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    main(args)