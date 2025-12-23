import argparse
import numpy as np
import os
from tqdm import tqdm
import sys

# Add project root to path to ensure imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.image import preprocess_frame

def main(args):
    print(f"Processing data from {args.input_dir}...")
    
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.npz')]
    os.makedirs(args.out_dir, exist_ok=True)
    
    if not files:
        print("No .npz files found!")
        return

    print(f"Found {len(files)} rollouts. Preprocessing...")
    
    for f in tqdm(files):
        # Load raw
        raw_path = os.path.join(args.input_dir, f)
        data = np.load(raw_path)
        raw_obs = data['obs'] # List of frames
        raw_actions = data['actions']
        raw_rewards = data['rewards']
        raw_dones = data['dones']
        
        # Process frames
        processed_frames = []
        for frame in raw_obs:
            processed_frames.append(preprocess_frame(frame, args.resize))
        
        # Save processed
        save_path = os.path.join(args.out_dir, f)
        np.savez_compressed(
            save_path,
            obs=np.array(processed_frames), # float32, (T, 3, 64, 64)
            actions=raw_actions,
            rewards=raw_rewards,
            dones=raw_dones
        )
        
    print(f"Preprocessed data saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resize", type=int, default=64)
    args = parser.parse_args()
    main(args)