import random
import torch
import os
import numpy as np
import config
import copy

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """
    Saves the model and optimizer state dicts to a checkpoint file.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Loads a checkpoint into the model and optimizer.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging.
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    """
    Sets the seed for all random number generators for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr_scheduler(optimizer, lr_scheduler_type='StepLR', step_size=10, gamma=0.1):
    """
    Returns the learning rate scheduler based on the provided type.
    """
    if lr_scheduler_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler_type == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif lr_scheduler_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    else:
        raise ValueError(f"Unknown scheduler type: {lr_scheduler_type}")


def get_device():
    """
    Returns the appropriate device ('cuda' if GPU is available, else 'cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
