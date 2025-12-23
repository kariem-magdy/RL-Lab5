"""
Utility functions for seeding and miscellaneous tasks.
"""
import random
import numpy as np
import torch
from src.utils.image import preprocess_frame # Re-export for backward compatibility if needed

def set_seed(seed):
    """Sets the seed for reproducibility across Python, Numpy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False