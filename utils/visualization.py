import os

import cv2


def apply_contrast_enhancement(image):
    """
    Improve local contrast using CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def generate_zoomed_view(image, zoom_ratio=0.6):
    """
    Generate a center-cropped zoomed view of the lung region.
    """
    h, w = image.shape
    crop_h = int(h * zoom_ratio)
    crop_w = int(w * zoom_ratio)

    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2

    cropped = image[start_y:start_y + crop_h,
                    start_x:start_x + crop_w]

    zoomed = cv2.resize(cropped, (w, h))
    return zoomed


def generate_binarized_image(image):
    """
    Generate a binarized CT image using Otsu thresholding.

    White  -> high-density regions
    Black  -> air / low-density regions

    NOTE: This is NOT cancer localization.
    """
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def save_visual_outputs(original, enhanced, contrast, zoomed, binary, output_dir):
    """
    Save all generated visual outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "original": os.path.join(output_dir, "original.png"),
        "enhanced": os.path.join(output_dir, "gan_enhanced.png"),
        "contrast": os.path.join(output_dir, "contrast_enhanced.png"),
        "zoomed": os.path.join(output_dir, "zoomed_view.png"),
        "binary": os.path.join(output_dir, "binarized.png")
    }

    cv2.imwrite(paths["original"], original)
    cv2.imwrite(paths["enhanced"], enhanced)
    cv2.imwrite(paths["contrast"], contrast)
    cv2.imwrite(paths["zoomed"], zoomed)
    cv2.imwrite(paths["binary"], binary)

    return paths
