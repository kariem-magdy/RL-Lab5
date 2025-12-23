"""
Centralized image processing utilities.
"""
import cv2
import numpy as np

def preprocess_frame(frame, resize_dim=64):
    """
    Resizes frame to (resize_dim, resize_dim) and normalizes to [0, 1].
    Input: (H, W, 3) uint8 or similar
    Output: (3, resize_dim, resize_dim) float32
    """
    if frame is None:
        return np.zeros((3, resize_dim, resize_dim), dtype=np.float32)
        
    # Resize
    frame = cv2.resize(frame, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
    # Normalize 0-1
    frame = frame.astype(np.float32) / 255.0
    # Channel First (C, H, W)
    frame = np.transpose(frame, (2, 0, 1))
    return frame