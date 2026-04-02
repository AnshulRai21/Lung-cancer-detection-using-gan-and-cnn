import cv2
import numpy as np


def resize_image(image: np.ndarray, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Resize input image to the expected model size."""
    return cv2.resize(image, size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]."""
    return image.astype(np.float32) / 255.0


def preprocess_image(image_path: str, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Load, grayscale, resize, normalize, and reshape CT image for model input."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError('Unable to read image file.')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = resize_image(gray, size)
    normalized = normalize_image(resized)

    # Output shape: (1, H, W, 1)
    return np.expand_dims(normalized, axis=(0, -1))
