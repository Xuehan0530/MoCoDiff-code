# MoCoDiff: Motion-Coherent Diffusion for Video Generation

Official PyTorch implementation of our paper.

## Overview

MoCoDiff is a diffusion-based video generation model that ensures temporal coherence across generated frames through dedicated temporal attention mechanisms integrated into a UNet backbone. The key contributions are:

- **Temporal Attention Blocks**: Cross-frame attention layers that enforce motion coherence.
- **Cosine Noise Schedule**: Improved noise schedule for stable training.
- **DDIM Sampling**: Fast deterministic inference via DDIM.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Python ≥ 3.9** and **PyTorch ≥ 2.0** are required.

## Project Structure

```
MoCoDiff-code/
├── configs/
│   └── default.yaml         # Default hyperparameters
├── datasets/
│   ├── __init__.py
│   └── video_dataset.py     # VideoDataset and DataLoader builder
├── models/
│   ├── __init__.py
│   ├── unet.py              # UNet backbone with temporal attention
│   ├── diffusion.py         # Gaussian diffusion process (DDPM/DDIM)
│   └── mocodiff.py          # Top-level MoCoDiff model
├── utils/
│   ├── __init__.py
│   ├── misc.py              # Checkpointing, logging, distributed helpers
│   └── visualization.py     # Video saving utilities
├── train.py                 # Training entry point
├── inference.py             # Inference / sample generation entry point
└── requirements.txt
```

## Data Preparation

Organize your dataset as a directory of video subdirectories, each containing sequentially named frame images:

```
data/videos/
    video_001/
        frame_0000.png
        frame_0001.png
        ...
    video_002/
        ...
```

Update `data.data_root` in `configs/default.yaml` to point to your dataset.

## Training

**Single GPU:**

```bash
python train.py --config configs/default.yaml
```

**Multi-GPU (using `torchrun`):**

```bash
torchrun --nproc_per_node=NUM_GPUS train.py --config configs/default.yaml
```

Training logs and checkpoints are saved to `outputs/mocodiff/` by default. TensorBoard logs can be viewed with:

```bash
tensorboard --logdir outputs/mocodiff/tensorboard
```

To resume from a checkpoint, set `training.resume` in `configs/default.yaml`:

```yaml
training:
  resume: "outputs/mocodiff/checkpoint_0500000.pt"
```

## Inference

Generate video samples from a trained checkpoint:

```bash
python inference.py \
  --config configs/default.yaml \
  --checkpoint outputs/mocodiff/checkpoint_final.pt \
  --sampler ddim \
  --ddim_steps 50 \
  --num_samples 16 \
  --output_dir outputs/samples
```

Generated videos are saved as `outputs/samples/generated_videos.mp4`.

## Configuration

All hyperparameters are controlled via `configs/default.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| model | model_channels | 128 | Base UNet channel width |
| model | channel_mult | [1,2,4,8] | Channel multipliers per resolution |
| model | num_frames | 16 | Frames per video clip |
| model | timesteps | 1000 | Diffusion timesteps |
| model | schedule | cosine | Noise schedule |
| training | lr | 1e-4 | Learning rate |
| training | num_steps | 500000 | Total training steps |
| training | ema_decay | 0.9999 | EMA decay rate |
| inference | sampler | ddim | Sampling method (ddpm/ddim) |
| inference | ddim_steps | 50 | DDIM denoising steps |
