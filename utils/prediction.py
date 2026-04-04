import os
import importlib
import importlib.util

import numpy as np

_PREDICTION_STATE = {
    'loaded': False,
    'model_path': None,
    'mode': 'heuristic-fallback',
    'error': None,
}
_PREDICTION_MODEL = None
_CANCER_CLASS_INDEX = int(os.getenv('CANCER_CLASS_INDEX', '0'))
_TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
torch = importlib.import_module('torch') if _TORCH_AVAILABLE else None
CNN = importlib.import_module('utils.classification').CNN if _TORCH_AVAILABLE else None


def load_prediction_model(model_path: str) -> dict:
    """
    Load binary cancer classifier if checkpoint exists.

    Expected checkpoint: state_dict for utils.classification.CNN.
    If loading fails, the app falls back to a lightweight heuristic.
    """
    global _PREDICTION_MODEL

    _PREDICTION_STATE['model_path'] = model_path
    _PREDICTION_STATE['loaded'] = False
    _PREDICTION_STATE['mode'] = 'heuristic-fallback'
    _PREDICTION_STATE['error'] = None
    _PREDICTION_MODEL = None

    if os.path.exists(model_path) and _TORCH_AVAILABLE and CNN is not None:
        try:
            model = CNN()
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            _PREDICTION_MODEL = model
            _PREDICTION_STATE['loaded'] = True
            _PREDICTION_STATE['mode'] = 'cnn-state-dict'
        except Exception as exc:
            _PREDICTION_STATE['error'] = str(exc)
    elif not _TORCH_AVAILABLE:
        _PREDICTION_STATE['error'] = 'torch not available in runtime'

    return _PREDICTION_STATE.copy()


def predict_cancer(enhanced_image: np.ndarray) -> tuple[str, float]:
    """
    Return prediction label and confidence.

    Uses trained CNN when available, otherwise falls back to heuristic.
    """
    if _PREDICTION_MODEL is not None:
        image = enhanced_image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = _PREDICTION_MODEL(tensor)

        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        label = 'Cancer' if pred.item() == _CANCER_CLASS_INDEX else 'No Cancer'
        return label, round(confidence.item() * 100.0, 2)

    # Heuristic fallback for environments without a compatible checkpoint.
    mean_intensity = float(np.mean(enhanced_image))
    score = max(0.0, min(1.0, (128.0 - mean_intensity) / 128.0))

    if score >= 0.5:
        label = 'Cancer'
        confidence = 50.0 + (score - 0.5) * 100.0
    else:
        label = 'No Cancer'
        confidence = 50.0 + (0.5 - score) * 100.0

    return label, round(confidence, 2)
