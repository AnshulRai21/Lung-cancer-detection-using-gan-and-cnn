import cv2
import numpy as np


def preprocess_image(image_path, size=(128, 128)):
    """
    Preprocess uploaded lung CT image.

    Steps:
    1. Read image
    2. Convert to grayscale
    3. Resize
    4. Normalize
    5. Expand dimensions for model input

    Returns:
        numpy array of shape (1, H, W, 1)
    """

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read image file")

    # Convert to grayscale (CT scans are intensity-based)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to model input size
    resized = cv2.resize(gray, size)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Shape -> (1, H, W, 1)
    preprocessed = np.expand_dims(normalized, axis=(0, -1))

    return preprocessed
