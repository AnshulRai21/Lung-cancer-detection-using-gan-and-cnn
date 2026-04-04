"""Dataset and augmentations for paired lung CT enhancement."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LungCTPairDataset(Dataset):
    """
    Paired dataset for cGAN training.

    Expected layout:
      root/
        input/  (degraded/noisy images)
        target/ (clean/enhanced images)

    If target image is missing, target defaults to input for demo compatibility.
    """

    def __init__(self, root_dir: str, image_size: int = 128, augment: bool = True):
        self.root = Path(root_dir)
        self.input_dir = self.root / 'input'
        self.target_dir = self.root / 'target'
        self.image_size = image_size
        self.augment = augment
        self.files = sorted(p for p in self.input_dir.glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})

    def __len__(self) -> int:
        return len(self.files)

    def _read_gray(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Failed to read image: {path}')
        return cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

    def _augment_pair(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return x, y

        if np.random.rand() < 0.5:
            x = cv2.flip(x, 1)
            y = cv2.flip(y, 1)

        if np.random.rand() < 0.3:
            angle = np.random.choice([-10, -5, 5, 10])
            center = (self.image_size // 2, self.image_size // 2)
            matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            x = cv2.warpAffine(x, matrix, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)
            y = cv2.warpAffine(y, matrix, (self.image_size, self.image_size), flags=cv2.INTER_LINEAR)

        return x, y

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        # Normalize CT to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(image).unsqueeze(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path = self.files[idx]
        target_path = self.target_dir / input_path.name

        input_img = self._read_gray(input_path)
        target_img = self._read_gray(target_path) if target_path.exists() else input_img.copy()

        input_img, target_img = self._augment_pair(input_img, target_img)
        return self._to_tensor(input_img), self._to_tensor(target_img)
