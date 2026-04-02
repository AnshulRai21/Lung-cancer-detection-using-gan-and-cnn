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

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── gan_pipeline/
│   ├── __init__.py
│   ├── architectures.py
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   └── train.py
├── utils/
│   ├── preprocessing.py
│   ├── enhancement.py
│   ├── visualization.py
│   └── prediction.py
├── templates/
│   ├── index.html
│   └── result.html
├── models/
└── static/
```

## Training the advanced GAN

Prepare paired dataset:

```text
your_dataset/
  input/
    img1.png
    img2.png
  target/
    img1.png
    img2.png
```

Run training:

```bash
python -m gan_pipeline.train \
  --data-dir your_dataset \
  --epochs 50 \
  --batch-size 8 \
  --image-size 128
```

Artifacts:
- Checkpoints: `checkpoints/advanced_gan_generator.pth`, `checkpoints/advanced_gan_discriminator.pth`
- Epoch visuals: `training_visuals/epoch_XXXX_input.png` and `epoch_XXXX_generated.png`

## Flask demo app

```bash
python app.py
```

Open: `http://127.0.0.1:5000/`

Pipeline in app: **Upload → Preprocess → Enhance → Visualize → Predict → Result**.

## Viva Notes

- The GAN training code is intentionally modular and explainable (`architectures`, `dataset`, `losses`, `metrics`, `train`).
- The model includes research-aligned stability and quality tricks while remaining feasible on limited hardware.
- You can later plug the trained `advanced_gan_generator.pth` into runtime inference for production-quality enhancement.
