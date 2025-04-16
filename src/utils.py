import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_directories(*paths):
    """
    Create directories if they don't exist.

    Args:
        *paths: Paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_model(model, path, optimizer=None, epoch=None, loss=None):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        path: Path to save model
        optimizer: Optional optimizer to save
        epoch: Optional epoch number
        loss: Optional loss value
    """
    state_dict = {
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        state_dict['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        state_dict['epoch'] = epoch

    if loss is not None:
        state_dict['loss'] = loss

    torch.save(state_dict, path)
    print(f"Model saved to {path}")


def load_model(model, path, optimizer=None, device='cpu'):
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        path: Path to load model from
        optimizer: Optional optimizer to load state
        device: Device to load model to

    Returns:
        model: Loaded model
        optimizer: Loaded optimizer (if provided)
        checkpoint: Loaded checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Model loaded from {path}")

    return model, optimizer, checkpoint

