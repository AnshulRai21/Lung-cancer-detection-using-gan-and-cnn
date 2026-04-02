# Lung CT Image Enhancement using GAN and Lung Cancer Detection

Flask-based capstone demo application for:
1. Lung CT image preprocessing
2. GAN-style image enhancement (with placeholder fallback)
3. Visualization outputs (contrast, zoomed lung region, binarized view)
4. Lung cancer prediction (model metadata + dummy fallback classifier)

> **Disclaimer:** This system assists medical professionals and does not replace diagnosis.

## Features
- Upload CT image from browser
- End-to-end pipeline:
  - Upload → Preprocess → GAN Enhance → Visualize → Predict → Result page
- Model metadata loaded once at startup
- Works in demo mode even when model runtimes are unavailable

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── templates/
│   ├── index.html
│   └── result.html
├── utils/
│   ├── preprocessing.py
│   ├── enhancement.py
│   ├── visualization.py
│   └── prediction.py
├── models/
│   ├── gan_generator.pth
│   └── cancer_classifier.pth
└── static/
    ├── uploads/
    └── outputs/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000/`

## Notes for Viva / Demo
- `utils/enhancement.py` and `utils/prediction.py` currently include fallback logic for smooth demos.
- Replace placeholder logic with real GAN/CNN inference as needed.
