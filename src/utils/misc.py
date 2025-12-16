"""
Utility functions for seeding and preprocessing.
"""
import random
import numpy as np
import torch
import cv2

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_frame(frame):
    """
    Resizes frame to 64x64 and normalizes to [0, 1].
    Input: (210, 160, 3) uint8
    Output: (3, 64, 64) float32
    """
    # Resize to 64x64
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    # Normalize to 0-1
    frame = frame.astype(np.float32) / 255.0
    # CHW format for PyTorch
    frame = np.transpose(frame, (2, 0, 1)) 
    return frame