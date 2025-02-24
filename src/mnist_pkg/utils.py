from typing import Optional
from pathlib import Path
import torch

from mnist_pkg.constants import MODELS_PATH

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
    

def save_model(model: torch.nn.Module, epoch: int, optimiser: torch.optim.Optimizer, loss: float, model_filename: str = "model.pth") -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "loss": loss,
    }
    MODELS_PATH.mkdir(exist_ok=True)
    torch.save(checkpoint, MODELS_PATH / model_filename)


def load_model(model: torch.nn.Module, optimiser: torch.optim.Optimizer, model_filename: str = "model.pth") -> tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    checkpoint = torch.load(MODELS_PATH / model_filename, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimiser, epoch, loss