import argparse
import yaml
import torch
import os
import sys
import numpy as np
import cma
import torch.nn.functional as F
import wandb
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.mdn_lstm import MDNLSTM
from src.models.controller import Controller
from src.models.vae import VAE
from src.utils.seed import set_seed
from src.utils.tracking import init_wandb

def validate_in_real_env(config, vae, lstm, controller, device, generation):
    """
    Runs a single episode in the REAL environment to visualize progress.
    Returns: Reward, Video Path, Duration (Steps)
    """
    video_subdir = os.path.join(config['video_dir'], f"training_gen_{generation}")
    os.makedirs(video_subdir, exist_ok=True)
    
    env = gym.make(config['env_name'], render_mode='rgb_array')
    env = RecordVideo(env, video_folder=video_subdir, episode_trigger=lambda x: True, name_prefix="eval")
    
    obs, _ = env.reset(seed=config['seed'])
    done = False
    truncated = False
    total_reward = 0
    
    h = torch.zeros(1, 1, config['lstm_hidden_dim']).to(device)
    c = torch.zeros(1, 1, config['lstm_hidden_dim']).to(device)
    hidden = (h, c)
    
    steps = 0
    while not (done or truncated) and steps < config['max_frames']:
        frame = cv2.resize(obs, (config['resize_dim'], config['resize_dim']))
        frame = torch.tensor(frame).float().permute(2, 0, 1) / 255.0
        frame_tensor = frame.unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(frame_tensor)
            z = mu 
            
            action = controller.get_action(z, hidden[0].squeeze(0))
            
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            action_one_hot = torch.zeros(1, 1, 4).to(device)
            action_one_hot[0, 0, action] = 1
            _, _, _, _, hidden = lstm(z.unsqueeze(0), action_one_hot, hidden)
            
        steps += 1
    
    env.close()
    
    video_path = None
    for f in os.listdir(video_subdir):
        if f.endswith(".mp4"):
            video_path = os.path.join(video_subdir, f)
            break
            
    return total_reward, video_path, steps

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
            k_indices = cat.sample()
            
            mu_sample = torch.gather(mu, 2, k_indices.unsqueeze(2)).squeeze(2)
            sigma_sample = torch.gather(sigma, 2, k_indices.unsqueeze(2)).squeeze(2)
            
            dist = torch.distributions.Normal(mu_sample, sigma_sample)
            z_next = dist.sample()
            
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
    
    vae_path = os.path.join(config['checkpoint_dir'], "vae_best.pth")
    lstm_path = os.path.join(config['checkpoint_dir'], "lstm_best.pth")
    
    if not os.path.exists(vae_path) or not os.path.exists(lstm_path):
        print("❌ Models not found.")
        sys.exit(1)

    vae = VAE(config['vae_latent_dim'], resize_dim=config['resize_dim']).to(device)
    vae.load_state_dict(torch.load(vae_path, map_location=device)['model_state_dict'])
    vae.eval()

    lstm = MDNLSTM(config['vae_latent_dim'], 4, config['lstm_hidden_dim'], config['lstm_num_gaussians']).to(device)
    lstm.load_state_dict(torch.load(lstm_path, map_location=device)['model_state_dict'])
    lstm.eval() 
    
    controller = Controller(config['vae_latent_dim'], config['lstm_hidden_dim']).to(device)
    
    n_params = controller.count_parameters()
    print(f"Training Controller in DREAM. Params: {n_params}")
    
    es = cma.CMAEvolutionStrategy(n_params * [0], 0.1, {'popsize': config['controller_pop_size']})
    
    best_reward = -np.inf
    
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
        
        log_data = {"dream/avg_reward": avg_r, "dream/max_reward": max_r}

        if gen % config.get('controller_eval_freq', 10) == 0:
            controller.set_parameters(es.result.xbest)
            print(f"🎥 Validating in Real Environment (Gen {gen})...")
            real_reward, video_file, real_duration = validate_in_real_env(config, vae, lstm, controller, device, gen)
            print(f"   -> Reward: {real_reward}, Duration: {real_duration}")
            
            log_data["real_eval/reward"] = real_reward
            log_data["real_eval/duration"] = real_duration
            if video_file and args.log:
                log_data["real_eval/video"] = wandb.Video(video_file, fps=30, format="mp4")

        if args.log:
            wandb.log(log_data)
            
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