#!/usr/bin/env python3
"""
train.py — Standalone training script for U-Net radio deconvolution.
Usage: python scripts/train.py --epochs 80 --batch_size 16 --lr 1e-4
"""
import argparse, os, time
from pathlib import Path

import h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# --- import model and losses from this package ---
# from models.unet import UNet
# from models.losses import CombinedLoss
# from data.dataset import RadioImageDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--data",       type=str,   default="data/processed/dataset.h5")
    p.add_argument("--ckpt_dir",   type=str,   default="models/checkpoints")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Training for {args.epochs} epochs  |  lr={args.lr}  |  bs={args.batch_size}")
    # ... (paste training loop here)
