# Google Colab Runbook (Train + Outputs) for this Project

This runbook gives copy-paste cells to train your advanced cGAN in Colab and download outputs.

---

## 1) Create a new Colab notebook

- Open: https://colab.research.google.com/
- Runtime -> **Change runtime type** -> Hardware accelerator: **GPU**

---

## 2) Clone repository

```python
!git clone https://github.com/AnshulRai21/lung-CT-image-enhancement-using-GAN-and-lung-cancer-detection-using-a-deep-learning-model.git
%cd lung-CT-image-enhancement-using-GAN-and-lung-cancer-detection-using-a-deep-learning-model
```

---

## 3) Install dependencies

```python
!pip install -r requirements.txt
```

If torch version conflicts in Colab runtime, run:

```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 4) Prepare dataset structure

Expected:

```text
your_dataset/
  input/
  target/
```

### Option A: Upload zip manually

```python
from google.colab import files
uploaded = files.upload()  # upload dataset.zip
```

```python
!unzip -q dataset.zip -d /content/
!ls /content/your_dataset
```

### Option B: Use Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set your path:

```python
DATA_DIR = "/content/drive/MyDrive/your_dataset"
```

---

## 5) Start training

```python
DATA_DIR = "/content/your_dataset"  # change if using drive path

!python -m gan_pipeline.train \
  --data-dir "$DATA_DIR" \
  --epochs 50 \
  --batch-size 8 \
  --image-size 128 \
  --visual-dir training_visuals \
  --checkpoint-dir checkpoints
```

This will generate:
- `training_visuals/epoch_XXXX_input.png`
- `training_visuals/epoch_XXXX_generated.png`
- `checkpoints/advanced_gan_generator.pth`
- `checkpoints/advanced_gan_discriminator.pth`

---

## 6) Download outputs

```python
!zip -r training_artifacts.zip training_visuals checkpoints
from google.colab import files
files.download('training_artifacts.zip')
```

---

## 7) (Optional) Run Flask app in Colab (for quick demo only)

```python
!python app.py
```

> Note: Colab is not ideal for persistent web hosting. Prefer local run or deployment platform for live demo.

---

## 8) Troubleshooting

- **CUDA OOM**: reduce `--batch-size` to 4 or 2.
- **Slow training**: reduce `--image-size` to 96.
- **Missing files**: verify `input/` and `target/` filenames match.
- **Low quality early epochs**: train longer; monitor PSNR/SSIM and visual outputs.

---

## 9) Suggested capstone settings (safe defaults)

```text
epochs: 30-50
batch-size: 4-8
image-size: 128
```

These defaults balance quality and Colab feasibility.
