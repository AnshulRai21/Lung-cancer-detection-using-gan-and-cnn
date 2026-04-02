import os

import cv2
import numpy as np

_GAN_STATE = {
    'loaded': False,
    'model_path': None,
    'mode': 'placeholder',
}


def load_gan_model(model_path: str) -> dict:
    """
    Load GAN model metadata.

    For this demo step, if the model file exists we mark it as loaded,
    otherwise we keep a placeholder mode so the app remains runnable.
    """
    _GAN_STATE['model_path'] = model_path
    _GAN_STATE['loaded'] = os.path.exists(model_path)
    _GAN_STATE['mode'] = 'file-available' if _GAN_STATE['loaded'] else 'placeholder'
    return _GAN_STATE.copy()


def enhance_image(preprocessed_image: np.ndarray) -> np.ndarray:
    """
    Generate enhanced image from preprocessed tensor.

    Input expected shape: (1, H, W, 1), normalized [0,1].
    Output: uint8 2D image.
    """
    image_2d = preprocessed_image.squeeze()
    image_u8 = np.clip(image_2d * 255.0, 0, 255).astype(np.uint8)

    # Placeholder enhancement: unsharp masking to improve local contrast.
    blurred = cv2.GaussianBlur(image_u8, (0, 0), 1.2)
    enhanced = cv2.addWeighted(image_u8, 1.6, blurred, -0.6, 0)

    return np.clip(enhanced, 0, 255).astype(np.uint8)
