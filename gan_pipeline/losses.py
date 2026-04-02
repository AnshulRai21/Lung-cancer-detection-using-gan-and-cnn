"""Losses for stable medical CT enhancement GAN training."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyFeatureExtractor(nn.Module):
    """Lightweight feature extractor for perceptual loss (hardware-friendly)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiObjectiveLoss(nn.Module):
    def __init__(self, lambda_l1: float = 100.0, lambda_perceptual: float = 10.0):
        super().__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()
        self.perc_criterion = nn.L1Loss()
        self.features = TinyFeatureExtractor()

        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual

    def adversarial_loss(self, pred: torch.Tensor, is_real: bool, smooth: float = 0.0) -> torch.Tensor:
        target_value = 1.0 - smooth if is_real else smooth
        target = torch.full_like(pred, target_value)
        return self.adv_criterion(pred, target)

    def generator_loss(self, d_fake_pred: torch.Tensor, fake: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        adv = self.adversarial_loss(d_fake_pred, is_real=True, smooth=0.1)
        l1 = self.l1_criterion(fake, target)
        perc = self.perc_criterion(self.features(fake), self.features(target))
        total = adv + self.lambda_l1 * l1 + self.lambda_perceptual * perc
        return {'total': total, 'adv': adv, 'l1': l1, 'perceptual': perc}
