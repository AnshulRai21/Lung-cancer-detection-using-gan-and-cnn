"""Image quality metrics for CT enhancement model evaluation."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def denorm_to_uint8(image: np.ndarray) -> np.ndarray:
    image = ((image + 1.0) * 127.5).clip(0, 255)
    return image.astype(np.uint8)


def compute_psnr_ssim(generated: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Compute PSNR and SSIM on single-channel images in [-1, 1] range."""
    gen_u8 = denorm_to_uint8(generated)
    tgt_u8 = denorm_to_uint8(target)
    psnr = peak_signal_noise_ratio(tgt_u8, gen_u8, data_range=255)
    ssim = structural_similarity(tgt_u8, gen_u8, data_range=255)
    return float(psnr), float(ssim)
