"""
Training script for MoCoDiff.

Usage (single GPU):
    python train.py --config configs/default.yaml

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=NUM_GPUS train.py --config configs/default.yaml
"""

import os
import argparse
import logging
import yaml
from copy import deepcopy

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from models import MoCoDiff
from datasets import build_dataloader
from utils import (
    set_seed,
    setup_logging,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    is_main_process,
    AverageMeter,
    save_video_grid,
)


def get_warmup_scheduler(optimizer, warmup_steps):
    """Linear warmup followed by constant learning rate."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def build_ema_model(model):
    """Create an EMA (exponential moving average) copy of the model."""
    ema = deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """Update EMA model parameters."""
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MoCoDiff")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training (set automatically by torchrun)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Distributed setup
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = train_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(log_dir=output_dir, rank=rank)
    logger = logging.getLogger(__name__)
    set_seed(train_cfg.get("seed", 42) + rank)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MoCoDiff(model_cfg).to(device)
    ema_model = build_ema_model(model).to(device)

    if is_main_process(rank):
        logger.info(f"Model parameters: {count_parameters(model):,}")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = build_dataloader(data_cfg, split="train")
    val_loader = build_dataloader(data_cfg, split="val")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=0.0)
    scheduler = get_warmup_scheduler(optimizer, train_cfg.get("lr_warmup_steps", 5000))
    scaler = GradScaler()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    resume_path = train_cfg.get("resume")
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer)
        logger.info(f"Resumed from step {start_step}: {resume_path}")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    if is_main_process(rank):
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

    # ── Training loop ─────────────────────────────────────────────────────────
    raw_model = model.module if is_distributed else model
    loss_meter = AverageMeter()
    data_iter = iter(train_loader)
    num_steps = train_cfg["num_steps"]
    log_every = train_cfg.get("log_every", 100)
    save_every = train_cfg.get("save_every", 5000)
    sample_every = train_cfg.get("sample_every", 2000)
    ema_decay = train_cfg.get("ema_decay", 0.9999)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    logger.info(f"Starting training from step {start_step}")
    model.train()

    for step in range(start_step, num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        video = batch["video"].to(device)  # (B, T, C, H, W)

        optimizer.zero_grad()
        with autocast():
            loss = model(video)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        update_ema(ema_model, raw_model, ema_decay)
        loss_meter.update(loss.item())

        if is_main_process(rank):
            if (step + 1) % log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step [{step+1}/{num_steps}] loss={loss_meter.avg:.4f} lr={lr:.2e}"
                )
                if writer:
                    writer.add_scalar("train/loss", loss_meter.avg, step + 1)
                    writer.add_scalar("train/lr", lr, step + 1)
                loss_meter.reset()

            if (step + 1) % save_every == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_{step+1:07d}.pt")
                save_checkpoint(
                    {
                        "step": step + 1,
                        "model": raw_model.state_dict(),
                        "ema": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint: {ckpt_path}")

            if (step + 1) % sample_every == 0:
                ema_model.eval()
                sample_batch = train_cfg.get("sample_batch_size", 4)
                cfg = model_cfg
                samples = ema_model.diffusion.ddim_sample(
                    shape=(sample_batch, cfg["num_frames"], cfg["in_channels"],
                           data_cfg["frame_size"][0], data_cfg["frame_size"][1]),
                    device=device,
                    ddim_steps=50,
                    show_progress=False,
                )
                video_path = os.path.join(output_dir, f"sample_{step+1:07d}.mp4")
                save_video_grid(samples, video_path)
                logger.info(f"Saved sample video: {video_path}")
                ema_model.train()

    if is_main_process(rank):
        final_ckpt = os.path.join(output_dir, "checkpoint_final.pt")
        save_checkpoint(
            {
                "step": num_steps,
                "model": raw_model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            final_ckpt,
        )
        logger.info(f"Training complete. Final checkpoint: {final_ckpt}")
        if writer:
            writer.close()

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
