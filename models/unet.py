import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with group normalization and time conditioning."""

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1, groups=32):
        super().__init__()
        groups = min(groups, in_channels)
        out_groups = min(groups, out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(out_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        scale_shift = self.time_mlp(time_emb)
        scale_shift = rearrange(scale_shift, "b c -> b c 1 1")
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.block1(x)
        h = h * (scale + 1) + shift
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block for spatial feature mixing."""

    def __init__(self, channels, num_heads=8, groups=32):
        super().__init__()
        groups = min(groups, channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, min(num_heads, channels), batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x, _ = self.attn(x, x, x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + residual


class TemporalAttentionBlock(nn.Module):
    """Temporal attention block for motion-coherent feature mixing across frames."""

    def __init__(self, channels, num_heads=8, groups=32):
        super().__init__()
        groups = min(groups, channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(channels, min(num_heads, channels), batch_first=True)

    def forward(self, x, num_frames):
        """
        Args:
            x: (B*T, C, H, W) spatial features.
            num_frames: T, number of frames.
        """
        b_t, c, h, w = x.shape
        b = b_t // num_frames

        residual = x
        x = self.norm(x)
        x = rearrange(x, "(b t) c h w -> (b h w) t c", b=b, t=num_frames)
        x, _ = self.attn(x, x, x)
        x = rearrange(x, "(b h w) t c -> (b t) c h w", b=b, h=h, w=w)
        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


def _build_level_blocks(in_ch, out_ch, time_emb_dim, num_res_blocks, use_attn, num_heads, dropout):
    """Build a sequence of residual (+ optional attention) blocks for one resolution level."""
    blocks = nn.ModuleList()
    for i in range(num_res_blocks):
        ch_in = in_ch if i == 0 else out_ch
        layer = nn.ModuleList([ResBlock(ch_in, out_ch, time_emb_dim, dropout)])
        if use_attn:
            layer.append(AttentionBlock(out_ch, num_heads))
            layer.append(TemporalAttentionBlock(out_ch, num_heads))
        blocks.append(layer)
    return blocks


def _apply_level_blocks(blocks, x, t_emb, num_frames):
    """Apply a sequence of (ResBlock [+ AttentionBlock + TemporalAttentionBlock]) groups."""
    for layer_group in blocks:
        for layer in layer_group:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, TemporalAttentionBlock):
                x = layer(x, num_frames)
            else:
                x = layer(x)
    return x


class UNet(nn.Module):
    """
    UNet backbone for MoCoDiff with temporal attention for motion coherence.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        model_channels: Base channel count.
        channel_mult: Channel multipliers for each resolution level.
        num_res_blocks: Number of residual blocks per resolution level.
        attention_resolutions: Downsampling factors at which to apply spatial attention.
        num_frames: Number of frames in the video clip.
        dropout: Dropout probability.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(2, 4),
        num_frames=16,
        dropout=0.1,
        num_heads=8,
    ):
        super().__init__()
        self.num_frames = num_frames
        time_emb_dim = model_channels * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = model_channels
        skip_channels = [ch]  # channels at each encoder output (for skip connections)

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            ds = 2 ** level
            use_attn = ds in attention_resolutions
            blocks = _build_level_blocks(
                ch, out_ch, time_emb_dim, num_res_blocks, use_attn, num_heads, dropout
            )
            self.encoder_blocks.append(blocks)
            ch = out_ch
            skip_channels.append(ch)

            if level < len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.middle_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.middle_attn = AttentionBlock(ch, num_heads)
        self.middle_temporal_attn = TemporalAttentionBlock(ch, num_heads)
        self.middle_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            skip_ch = skip_channels.pop()
            ds = 2 ** level
            use_attn = ds in attention_resolutions
            blocks = _build_level_blocks(
                ch + skip_ch, out_ch, time_emb_dim, num_res_blocks, use_attn, num_heads, dropout
            )
            self.decoder_blocks.append(blocks)
            ch = out_ch

            if level > 0:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        """
        Args:
            x: (B, T, C, H, W) noisy video frames.
            t: (B,) diffusion timesteps.

        Returns:
            (B, T, C, H, W) predicted noise.
        """
        b, num_frames, c, h, w = x.shape
        t_emb = self.time_embed(t)
        t_emb_frames = repeat(t_emb, "b d -> (b t) d", t=num_frames)

        # Merge batch and frame dimensions for spatial convolutions
        x = rearrange(x, "b t c h w -> (b t) c h w")
        h_feat = self.input_proj(x)

        # ── Encoder ───────────────────────────────────────────────────────────
        skips = [h_feat]
        for level_blocks, down in zip(self.encoder_blocks, self.downsamples):
            h_feat = _apply_level_blocks(level_blocks, h_feat, t_emb_frames, num_frames)
            skips.append(h_feat)
            h_feat = down(h_feat)

        # ── Bottleneck ────────────────────────────────────────────────────────
        h_feat = self.middle_block1(h_feat, t_emb_frames)
        h_feat = self.middle_attn(h_feat)
        h_feat = self.middle_temporal_attn(h_feat, num_frames)
        h_feat = self.middle_block2(h_feat, t_emb_frames)

        # ── Decoder ───────────────────────────────────────────────────────────
        for level_blocks, up in zip(self.decoder_blocks, self.upsamples):
            skip = skips.pop()
            h_feat = torch.cat([h_feat, skip], dim=1)
            h_feat = _apply_level_blocks(level_blocks, h_feat, t_emb_frames, num_frames)
            h_feat = up(h_feat)

        # Output projection
        out = self.output_proj(h_feat)
        return rearrange(out, "(b t) c h w -> b t c h w", b=b, t=num_frames)
