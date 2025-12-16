import sys, os, yaml, torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.vae import VAE
from src.models.mdn_rnn import MDNRNN
from src.models.controller import Controller
from src.utils.misc import preprocess_frame

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])
    
    # Load Models
    vae = VAE(config['vae_latent_dim']).to(device)
    vae.load_state_dict(torch.load("vae.pth", map_location=device))
    
    rnn = MDNRNN(config['vae_latent_dim'], 4, config['rnn_hidden_dim']).to(device)
    rnn.load_state_dict(torch.load("rnn.pth", map_location=device))
    
    controller = Controller(config['vae_latent_dim'], config['rnn_hidden_dim']).to(device)
    controller.load_state_dict(torch.load("controller.pth", map_location=device))
    
    # Setup Video Env
    env = gym.make(config['env_name'], render_mode='rgb_array')
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)
    
    print("Recording evaluation episode...")
    obs, _ = env.reset()
    done = False
    
    h = torch.zeros(1, 1, config['rnn_hidden_dim']).to(device)
    c = torch.zeros(1, 1, config['rnn_hidden_dim']).to(device)
    hidden = (h, c)
    
    while not done:
        frame = preprocess_frame(obs)
        frame = torch.tensor(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mu, _ = vae.encode(frame) # Use mean for determinstic eval
            z = mu
            action = controller.get_action(z, hidden[0].squeeze(0))
            
            obs, _, done, _, _ = env.step(action)
            
            action_one_hot = torch.zeros(1, 1, 4).to(device)
            action_one_hot[0, 0, action] = 1
            _, _, _, hidden = rnn(z.unsqueeze(0), action_one_hot, hidden)

    env.close()
    print("Video saved to videos/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)