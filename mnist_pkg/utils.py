from typing import Optional
import torch

def set_device(override: Optional[str] = None) -> torch.device:
    if override:
        return torch.device(override)
    
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because PyTorch was not built with MPS enabled")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device")
        return torch.device("cpu")
    else:
        return torch.device("mps")
    
