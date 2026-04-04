import os

import numpy as np

_PREDICTION_STATE = {
    'loaded': False,
    'model_path': None,
    'mode': 'dummy-fallback',
}


def load_prediction_model(model_path: str) -> dict:
    """
    Load classification model metadata.

    For this demo, if a model file exists we mark it as available.
    Prediction still uses a lightweight fallback heuristic so the app is
    runnable without deep-learning dependencies.
    """
    _PREDICTION_STATE['model_path'] = model_path
    _PREDICTION_STATE['loaded'] = os.path.exists(model_path)
    _PREDICTION_STATE['mode'] = 'file-available' if _PREDICTION_STATE['loaded'] else 'dummy-fallback'
    return _PREDICTION_STATE.copy()


def predict_cancer(enhanced_image: np.ndarray) -> tuple[str, float]:
    """
    Return a demo prediction label and confidence.

    Heuristic: lower mean intensity trends toward "Cancer" for demo only.
    """
    mean_intensity = float(np.mean(enhanced_image))
    score = max(0.0, min(1.0, (128.0 - mean_intensity) / 128.0))

    if score >= 0.5:
        label = 'Cancer'
        confidence = 50.0 + (score - 0.5) * 100.0
    else:
        label = 'No Cancer'
        confidence = 50.0 + (0.5 - score) * 100.0

    return label, round(confidence, 2)
