# Lung CT Image Enhancement using Advanced GAN + Cancer Detection Demo

This project now includes a **research-inspired conditional GAN (cGAN/Pix2Pix-style)** training pipeline for lung CT enhancement, plus a Flask demo app for inference visualization.

> **Disclaimer:** This system assists medical professionals and does not replace diagnosis.

## ✅ What was upgraded beyond baseline DCGAN

- **Conditional GAN design (Pix2Pix style):** generator takes CT input and predicts enhanced CT output.
- **Attention mechanism:** self-attention block in generator bottleneck.
- **Residual learning:** ResNet-style residual blocks to preserve medical structure.
- **Multi-loss objective:** adversarial loss + L1 loss + perceptual loss.
- **Training stabilization:** batch normalization, spectral normalization (discriminator), label smoothing.
- **Post-processing:** CLAHE + normalization after generator output.
- **Per-epoch evidence:** saves original + generated samples each epoch.
- **Quality metrics:** PSNR and SSIM reported during training.
- **Data augmentation:** random flip + mild rotations; CT intensity normalized to `[-1, 1]`.

These changes are implemented in `gan_pipeline/` and are intended to be demonstrably stronger and more stable than a basic GAN baseline.
# Lung CT Image Enhancement and Cancer Detection (Step 1)

This is the initial setup for a Flask-based capstone project.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── templates/
├── static/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── outputs/
├── utils/
├── models/
└── uploads/
```

## Setup Instructions

1. Create and activate a virtual environment:
   - Linux/macOS:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open in browser:
   - http://127.0.0.1:5000/

---

> Disclaimer: This system assists medical professionals and does not replace diagnosis.
