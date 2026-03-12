import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class VideoDataset(Dataset):
    """
    Dataset for loading video clips from a directory of frames or video files.

    Expects the data directory to be organized as:
        data_root/
            video_001/
                frame_0000.png
                frame_0001.png
                ...
            video_002/
                ...

    Args:
        data_root: Path to the root directory containing video subdirectories.
        num_frames: Number of frames to sample per clip.
        frame_size: Spatial resolution (height, width) to resize frames to.
        split: 'train' or 'val'.
        val_split: Fraction of data to use for validation.
        seed: Random seed for reproducible train/val splits.
        frame_ext: File extension for frame images.
    """

    def __init__(
        self,
        data_root,
        num_frames=16,
        frame_size=(256, 256),
        split="train",
        val_split=0.1,
        seed=42,
        frame_ext="*.png",
    ):
        self.data_root = data_root
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.split = split

        # Discover all video directories
        self.video_dirs = sorted(
            d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)
        )

        if not self.video_dirs:
            raise RuntimeError(f"No video directories found in {data_root}")

        # Train/val split
        rng = random.Random(seed)
        rng.shuffle(self.video_dirs)
        n_val = max(1, int(len(self.video_dirs) * val_split))

        if split == "train":
            self.video_dirs = self.video_dirs[n_val:]
        elif split == "val":
            self.video_dirs = self.video_dirs[:n_val]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Build index: list of (video_dir, frame_paths)
        self.samples = []
        for vdir in self.video_dirs:
            frames = sorted(glob.glob(os.path.join(vdir, frame_ext)))
            if len(frames) >= num_frames:
                self.samples.append(frames)

        if not self.samples:
            raise RuntimeError(
                f"No videos with at least {num_frames} frames found in {data_root}"
            )

        # Define transforms
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize(frame_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(frame_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]

        # Randomly sample a contiguous clip
        max_start = len(frame_paths) - self.num_frames
        start = random.randint(0, max_start)
        selected = frame_paths[start : start + self.num_frames]

        # Load and transform frames
        frames = []
        for path in selected:
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))

        # Stack to (T, C, H, W)
        video = torch.stack(frames, dim=0)
        return {"video": video, "path": frame_paths[0]}


def build_dataloader(config, split="train"):
    """
    Build a DataLoader for video data.

    Args:
        config: Configuration dict containing dataset parameters.
        split: 'train' or 'val'.

    Returns:
        A DataLoader instance.
    """
    dataset = VideoDataset(
        data_root=config["data_root"],
        num_frames=config.get("num_frames", 16),
        frame_size=tuple(config.get("frame_size", [256, 256])),
        split=split,
        val_split=config.get("val_split", 0.1),
        seed=config.get("seed", 42),
        frame_ext=config.get("frame_ext", "*.png"),
    )

    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=(split == "train"),
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=(split == "train"),
    )

    return loader
