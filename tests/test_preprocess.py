import sys, os
import numpy as np
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.preprocess import preprocess_frame

def test_preprocess_shape_and_norm():
    # Random RGB frame 210x160
    dummy = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
    processed = preprocess_frame(dummy, resize_dim=64)
    
    assert processed.shape == (3, 64, 64)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0