import os
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir=None, rank=0):
    """Configure logging for training."""
    handlers = [logging.StreamHandler()]
    if log_dir is not None and rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, "train.log")))

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def count_parameters(model):
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, path):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, strict=True):
    """
    Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state.
        strict: Whether to strictly enforce parameter name matching.

    Returns:
        The step/epoch number stored in the checkpoint, or 0.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("step", 0)


def setup_distributed():
    """Initialize distributed training with NCCL backend."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def is_main_process(rank=None):
    """Return True if this is the main (rank-0) process."""
    if rank is not None:
        return rank == 0
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class AverageMeter:
    """Computes and stores the running average of a scalar value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
