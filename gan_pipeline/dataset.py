import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class LungCTPairDataset(Dataset):
    def __init__(self, data_dir, image_size=256, augment=True):
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment

        # 🔥 Load ALL images from folder + subfolders
        self.files = []
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    full_path = os.path.join(root, f)
                    self.files.append(full_path)

        print("Total images found:", len(self.files))

    def __len__(self):
        return len(self.files)

    def _read_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Safety check (if image fails to load)
        if img is None:
            raise ValueError(f"Error loading image: {path}")

        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img / 255.0  # normalize (0–1)
        img = np.expand_dims(img, axis=0)  # (1, H, W)

        return img

    def __getitem__(self, idx):
        input_path = self.files[idx]

        # 🔥 Use SAME image as target (no target_dir needed)
        target_path = input_path

        input_img = self._read_gray(input_path)
        target_img = self._read_gray(target_path)

        input_img = torch.tensor(input_img, dtype=torch.float32)
        target_img = torch.tensor(target_img, dtype=torch.float32)

        return input_img, target_img
