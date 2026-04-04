# Final Review (Code-Accurate) — Lung CT Enhancement and Detection Pipeline

This document is aligned strictly with the current codebase and is designed for final project review / viva.

## 1) Abstract

This project implements a Flask-based lung CT demo pipeline and a separate research-inspired GAN training stack.

- Web pipeline: Upload CT image → preprocessing → enhancement → visualization → prediction → result UI.
- Training pipeline: conditional GAN (Pix2Pix-style) with residual blocks, attention, multi-loss optimization, stabilization tricks, and image-quality metrics.

The design goal is to improve CT image quality while keeping the system runnable for capstone demos.

## 2) Introduction

Lung CT image quality has direct impact on downstream cancer assessment. Baseline GANs (e.g., plain DCGAN) often suffer from artifacts, unstable training, and less controllable outputs. This project addresses that by using conditional generation with explicit image-to-image mapping and quality-aware losses.

## 3) Algorithms / Methodology

### 3.1 Conditional GAN Architecture

Implemented in `gan_pipeline/architectures.py`:

- `AttentionResUNetGenerator`:
  - Encoder-decoder (Pix2Pix style)
  - Skip connections
  - Residual blocks (`ResidualBlock`)
  - Self-attention (`SelfAttention`) in bottleneck
- `PatchDiscriminator`:
  - Patch-level discrimination
  - Spectral normalization for stability

### 3.2 Multi-Loss Objective

Implemented in `gan_pipeline/losses.py`:

- Adversarial loss: `BCEWithLogitsLoss`
- Pixel loss: `L1Loss`
- Perceptual loss: feature-space L1 using `TinyFeatureExtractor`
- Combined objective:
  - `total = adv + lambda_l1 * l1 + lambda_perceptual * perceptual`

### 3.3 Training Stabilization

Used in `gan_pipeline/architectures.py` and `gan_pipeline/train.py`:

- Batch normalization in generator/discriminator blocks
- Spectral normalization in discriminator
- Label smoothing via adversarial targets for real/fake
- Adam optimizer with GAN-standard betas `(0.5, 0.999)`

### 3.4 Image Quality Post-processing

In training (`gan_pipeline/train.py`):

- CLAHE (`cv2.createCLAHE`) applied batch-wise to generated outputs
- Intensity normalized back to `[-1, 1]`

### 3.5 Evaluation Metrics

In `gan_pipeline/metrics.py` and `gan_pipeline/train.py`:

- PSNR
- SSIM
- Epoch-level averaged reporting during training

### 3.6 Dataset Handling

In `gan_pipeline/dataset.py`:

- Paired dataset structure:
  - `input/` (degraded/noisy)
  - `target/` (clean/enhanced)
- Augmentation:
  - random horizontal flip
  - small-angle random rotations
- CT normalization to `[-1, 1]`

## 4) End-to-End Demo Flow (Flask App)

Implemented in `app.py` and templates:

1. User uploads CT image (`templates/index.html`)
2. Backend validates file extension and saves upload
3. Preprocess image (`utils/preprocessing.py`)
4. Enhance image (`utils/enhancement.py` fallback enhancement)
5. Generate visualization outputs (`utils/visualization.py`)
6. Predict label/confidence (`utils/prediction.py` fallback classifier)
7. Render outputs + prediction (`templates/result.html`)

## 5) Results Section (What to Show in Poster)

Use these artifacts during demo/training review:

- Per-epoch GAN visuals from `training_visuals/`
  - `epoch_XXXX_input.png`
  - `epoch_XXXX_generated.png`
- Training console logs with:
  - Generator loss
  - Discriminator loss
  - PSNR
  - SSIM
- Web result page outputs:
  - Original
  - GAN Enhanced
  - Contrast Enhanced
  - Zoomed Region
  - Binarized View
  - Prediction + confidence

## 6) Conclusion

The current implementation includes advanced, research-aligned GAN training components and a full deployable demo flow. It is modular and viva-friendly, and can be further extended by replacing fallback inference with fully trained deployment models.

## 7) References (Conceptual)

- Pix2Pix / conditional GAN image-to-image paradigms
- PatchGAN discriminators
- Residual learning (ResNet blocks)
- Attention in generative models
- SSIM/PSNR as medical imaging quality metrics

---

## Important Accuracy Notes

- The Flask runtime enhancement and prediction modules currently include lightweight fallback logic for demo robustness.
- The advanced training pipeline is implemented and ready to train/export improved generator checkpoints.
- Any claim like “+5% classification accuracy” must be backed by your dataset-specific experiment logs.
