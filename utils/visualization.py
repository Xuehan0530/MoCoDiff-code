import torch
import numpy as np
import imageio
from torchvision.utils import make_grid


def tensor_to_image(tensor):
    """
    Convert a tensor in [-1, 1] to a uint8 NumPy array in [0, 255].

    Args:
        tensor: (..., C, H, W) float tensor.

    Returns:
        (..., H, W, C) uint8 NumPy array.
    """
    tensor = tensor.detach().cpu().float()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    tensor = tensor * 255
    return tensor.byte().numpy()


def save_video(frames, path, fps=8):
    """
    Save a video clip to disk.

    Args:
        frames: (T, H, W, C) uint8 NumPy array or (T, C, H, W) float tensor.
        path: Output file path (e.g., 'output.mp4').
        fps: Frames per second.
    """
    if isinstance(frames, torch.Tensor):
        frames = tensor_to_image(frames.permute(0, 2, 3, 1))  # (T, H, W, C)
    writer = imageio.get_writer(path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def save_video_grid(videos, path, fps=8, nrow=4):
    """
    Save a grid of video clips to a single video file.

    Args:
        videos: (B, T, C, H, W) float tensor in [-1, 1].
        path: Output file path.
        fps: Frames per second.
        nrow: Number of videos per row in the grid.
    """
    b, t, c, h, w = videos.shape
    frames_out = []
    for ti in range(t):
        grid = make_grid(videos[:, ti], nrow=nrow, normalize=True, value_range=(-1, 1))
        grid = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames_out.append(grid)

    writer = imageio.get_writer(path, fps=fps)
    for frame in frames_out:
        writer.append_data(frame)
    writer.close()
