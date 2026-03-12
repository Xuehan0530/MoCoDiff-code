"""
Inference script for MoCoDiff.

Usage:
    python inference.py --config configs/default.yaml --checkpoint outputs/mocodiff/checkpoint_final.pt
"""

import os
import argparse
import yaml

import torch

from models import MoCoDiff
from utils import load_checkpoint, save_video_grid, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MoCoDiff Inference")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint. Overrides config value."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save generated videos. Overrides config value."
    )
    parser.add_argument(
        "--sampler", type=str, default=None,
        choices=["ddpm", "ddim"],
        help="Sampling method. Overrides config value."
    )
    parser.add_argument(
        "--ddim_steps", type=int, default=None,
        help="Number of DDIM steps. Overrides config value."
    )
    parser.add_argument(
        "--ddim_eta", type=float, default=None,
        help="DDIM eta (0=deterministic, 1=DDPM). Overrides config value."
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Total number of samples to generate. Overrides config value."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Samples per batch. Overrides config value."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--use_ema", action="store_true", default=True,
        help="Load EMA weights from checkpoint (default: True)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    infer_cfg = config.get("inference", {})
    model_cfg = config["model"]

    # CLI overrides
    checkpoint = args.checkpoint or infer_cfg.get("checkpoint")
    output_dir = args.output_dir or infer_cfg.get("output_dir", "outputs/samples")
    sampler = args.sampler or infer_cfg.get("sampler", "ddim")
    ddim_steps = args.ddim_steps or infer_cfg.get("ddim_steps", 50)
    ddim_eta = args.ddim_eta if args.ddim_eta is not None else infer_cfg.get("ddim_eta", 0.0)
    num_samples = args.num_samples or infer_cfg.get("num_samples", 16)
    batch_size = args.batch_size or infer_cfg.get("batch_size", 4)
    num_frames = infer_cfg.get("num_frames", model_cfg.get("num_frames", 16))
    height = infer_cfg.get("height", 256)
    width = infer_cfg.get("width", 256)

    if checkpoint is None:
        raise ValueError("No checkpoint specified. Use --checkpoint or set inference.checkpoint in config.")

    os.makedirs(output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = MoCoDiff(model_cfg).to(device)

    ckpt = torch.load(checkpoint, map_location="cpu")
    key = "ema" if (args.use_ema and "ema" in ckpt) else "model"
    model.load_state_dict(ckpt[key])
    print(f"Loaded {'EMA' if key == 'ema' else 'model'} weights from: {checkpoint}")

    model.eval()

    # ── Generate samples ──────────────────────────────────────────────────────
    generated = []
    remaining = num_samples

    with torch.no_grad():
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            shape = (current_batch, num_frames, model_cfg.get("in_channels", 3), height, width)

            if sampler == "ddim":
                samples = model.diffusion.ddim_sample(
                    shape=shape,
                    device=device,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
            else:
                samples = model.diffusion.p_sample_loop(
                    shape=shape,
                    device=device,
                )

            generated.append(samples.cpu())
            remaining -= current_batch
            print(f"Generated {num_samples - remaining}/{num_samples} samples.")

    # Save results
    all_samples = torch.cat(generated, dim=0)  # (N, T, C, H, W)
    out_path = os.path.join(output_dir, "generated_videos.mp4")
    save_video_grid(all_samples, out_path, nrow=min(4, num_samples))
    print(f"Saved {num_samples} generated video(s) to: {out_path}")


if __name__ == "__main__":
    main()
