import torch
import torch.nn as nn
from .unet import UNet
from .diffusion import GaussianDiffusion


class MoCoDiff(nn.Module):
    """
    MoCoDiff: Motion-Coherent Diffusion Model for Video Generation.

    This model extends Gaussian diffusion with temporal attention mechanisms
    to generate temporally coherent video sequences.

    Args:
        config: A dictionary or object containing model hyperparameters.
            - in_channels (int): Input channels per frame. Default: 3.
            - out_channels (int): Output channels per frame. Default: 3.
            - model_channels (int): Base UNet channel count. Default: 128.
            - channel_mult (tuple): Channel multipliers. Default: (1, 2, 4, 8).
            - num_res_blocks (int): Residual blocks per level. Default: 2.
            - attention_resolutions (tuple): Spatial resolutions for attention. Default: (16, 8).
            - num_frames (int): Number of video frames. Default: 16.
            - dropout (float): Dropout probability. Default: 0.1.
            - num_heads (int): Number of attention heads. Default: 8.
            - timesteps (int): Diffusion timesteps. Default: 1000.
            - schedule (str): Noise schedule ('cosine' or 'linear'). Default: 'cosine'.
            - loss_type (str): Loss type ('l2' or 'l1'). Default: 'l2'.
    """

    def __init__(self, config):
        super().__init__()

        unet = UNet(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            model_channels=config.get("model_channels", 128),
            channel_mult=tuple(config.get("channel_mult", [1, 2, 4, 8])),
            num_res_blocks=config.get("num_res_blocks", 2),
            attention_resolutions=tuple(config.get("attention_resolutions", [16, 8])),
            num_frames=config.get("num_frames", 16),
            dropout=config.get("dropout", 0.1),
            num_heads=config.get("num_heads", 8),
        )

        self.diffusion = GaussianDiffusion(
            model=unet,
            timesteps=config.get("timesteps", 1000),
            schedule=config.get("schedule", "cosine"),
            loss_type=config.get("loss_type", "l2"),
        )

    def forward(self, x, t=None, noise=None):
        """Compute diffusion training loss."""
        return self.diffusion(x, t=t, noise=noise)

    @torch.no_grad()
    def sample(self, batch_size, num_frames, height, width, device, sampler="ddpm", **kwargs):
        """
        Generate video samples.

        Args:
            batch_size: Number of videos to generate.
            num_frames: Number of frames per video.
            height: Frame height.
            width: Frame width.
            device: Torch device.
            sampler: Sampling method ('ddpm' or 'ddim').
            **kwargs: Additional arguments passed to the sampler.

        Returns:
            Generated video frames as (B, T, C, H, W) tensor in [-1, 1].
        """
        shape = (batch_size, num_frames, self.diffusion.model.input_proj.in_channels, height, width)

        if sampler == "ddpm":
            return self.diffusion.p_sample_loop(shape, device, **kwargs)
        elif sampler == "ddim":
            return self.diffusion.ddim_sample(shape, device, **kwargs)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
