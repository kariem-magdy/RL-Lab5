import argparse
import numpy as np
import os
import cv2
import yaml
from tqdm import tqdm

def preprocess_frame(frame, resize_dim=64):
    """
    Input: (210, 160, 3) uint8
    Output: (3, 64, 64) float32 normalized
    """
    # Resize
    frame = cv2.resize(frame, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
    # Normalize 0-1
    frame = frame.astype(np.float32) / 255.0
    # Channel First (C, H, W)
    frame = np.transpose(frame, (2, 0, 1))
    return frame

def main(args):
    print(f"Loading raw data from {args.input}...")
    data = np.load(args.input, allow_pickle=True)
    raw_obs = data['obs']
    raw_actions = data['actions']
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Preprocessing frames...")
    count = 0
    for i in tqdm(range(len(raw_obs))):
        episode_frames = raw_obs[i]
        processed_frames = []
        for frame in episode_frames:
            processed_frames.append(preprocess_frame(frame, args.resize))
        
        # Save individual episodes to allow lazy loading later
        np.savez_compressed(
            os.path.join(args.out_dir, f"episode_{i}.npz"),
            obs=np.array(processed_frames),
            actions=raw_actions[i]
        )
        count += 1
        
    print(f"Preprocessed {count} episodes into {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resize", type=int, default=64)
    args = parser.parse_args()
    main(args)