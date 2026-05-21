# U-Net Radio Deconvolution

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abubakryagob/unet-radio-deconv/blob/main/notebooks/unet-radio-deconv.ipynb)

Deep-learning-based PSF deconvolution for radio interferometric images, implemented
as part of an ESO/SKAO-aligned research project.
A U-Net maps **dirty images → clean sky models**, learning to undo the smearing caused
by incomplete uv-plane coverage — replacing the classical CLEAN algorithm.

## Method

| Step | Detail |
|------|--------|
| Sky models | Random point sources + extended Gaussian blobs (256 × 256) |
| PSF simulation | ALMA-like uv-coverage via random baseline sampling, 2D FFT |
| Architecture | 4-level U-Net (PyTorch), MSE + SSIM loss, Adam + ReduceLROnPlateau |
| Baseline | Hogbom CLEAN (Python implementation) |
| Metrics | SSIM, PSNR on held-out test set (750 images) |

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/unet-radio-deconv
cd unet-radio-deconv
pip install -r requirements.txt
python scripts/train.py --epochs 80
python scripts/predict.py --input my_dirty.fits --checkpoint models/checkpoints/best_model.pth
```

Or run everything interactively in Google Colab (badge above).

## Results

| Method | SSIM ↑ | PSNR dB ↑ |
|--------|--------|-----------|
| U-Net  | ~0.99  | ~45 dB    |
| CLEAN  | ~0.58  | ~23 dB    |

*Results on 750-image test set; vary with training run.*

## Repository layout

unet-radio-deconv/
├── data/
│   ├── raw/                   # (empty — generated locally)
│   └── processed/dataset.h5   # HDF5 image pairs
├── models/
│   ├── checkpoints/           # .pth files
│   └── unet.py                # UNet class
├── notebooks/
│   └── 01_setup_and_data.ipynb
├── scripts/
│   ├── train.py
│   └── predict.py
├── results/
│   └── figures/               # loss curves, comparison plots
├── requirements.txt
└── README.md


## Citation / acknowledgement

If you use this code, please cite or acknowledge the ESO/SKAO-funded project it
supports, and the foundational U-Net paper:
> Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation", arXiv:1505.04597

## Author

Dr. Abubakr Y.A. Ibrahim — Institute of Space Sciences (ICE-CSIC), Barcelona
[abubakryagob.com](https://www.abubakryagob.com)
