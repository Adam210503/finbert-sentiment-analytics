import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Globally locks random seeds across Python, NumPy, and PyTorch (CPU, CUDA, and MPS).
    Modified from: https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    """
    # 1. Core Python and Environment
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Mathematical Computing
    np.random.seed(seed)
    
    # 3. Base PyTorch Engine
    torch.manual_seed(seed)
    
    # 4. Apple Silicon Graphics Backend (Crucial for your MacBook's MPS runner)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    # 5. NVIDIA CUDA Core (For portability if you push this to a cloud GPU server later)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # FIX FROM GIST COMMENTS:
        # Forces cuDNN to select algorithms deterministically
        torch.backends.cudnn.deterministic = True
        # Setting this to False prevents cuDNN from continuously benchmarking 
        # different convolution algorithms, which introduces random variance.
        torch.backends.cudnn.benchmark = False 

# Execute the seed freeze immediately
seed_everything(42)