#!/usr/bin/env python3
"""
predict.py — Run U-Net deconvolution on a single dirty image.
Usage:
    python scripts/predict.py --input dirty.png  --checkpoint models/checkpoints/best_model.pth
    python scripts/predict.py --input dirty.fits --checkpoint models/checkpoints/best_model.pth
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Try to load image ────────────────────────────────────────────────────────
def load_image(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in ('.fits', '.fit'):
        from astropy.io import fits
        data = fits.getdata(p).astype(np.float32)
        data = data.squeeze()          # drop degenerate axes (STOKES, FREQ)
    else:
        from skimage import io, transform, img_as_float32
        data = img_as_float32(io.imread(p, as_gray=True))
    # Normalise to [0, 1]
    if data.max() > 0:
        data /= data.max()
    return data


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    # from models.unet import UNet
    # model = UNet().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    # model.load_state_dict(ckpt["model_state_dict"])
    # model.eval()

    img   = load_image(args.input)       # (H, W) float32
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    with torch.no_grad():
        out = model(tensor).squeeze().cpu().numpy()

    # Save figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img, cmap="inferno", origin="lower"); ax1.set_title("Dirty"); ax1.axis("off")
    ax2.imshow(out, cmap="inferno", origin="lower"); ax2.set_title("U-Net clean"); ax2.axis("off")
    out_path = Path(args.input).stem + "_unet.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True)
    p.add_argument("--checkpoint", required=True)
    predict(p.parse_args())
