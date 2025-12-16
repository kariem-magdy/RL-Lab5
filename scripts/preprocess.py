import argparse
import numpy as np
import os
import cv2
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
        
        # Process frames
        processed_frames = []
        for frame in raw_obs:
            processed_frames.append(preprocess_frame(frame, args.resize))
        
        # Save processed
        save_path = os.path.join(args.out_dir, f)
        np.savez_compressed(
            save_path,
            obs=np.array(processed_frames), # float32, (T, 3, 64, 64)
            actions=raw_actions
        )
        
    print(f"Preprocessed data saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resize", type=int, default=64)
    args = parser.parse_args()
    main(args)