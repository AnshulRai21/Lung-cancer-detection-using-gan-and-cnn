"""Research-inspired conditional GAN architectures for lung CT enhancement."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).reshape(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).reshape(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).reshape(b, -1, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, h, w)
        return self.gamma * out + x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.block(x) + x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionResUNetGenerator(nn.Module):
    """Pix2Pix-style conditional generator with residual and attention modules."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 64):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.down1 = DownBlock(base, base * 2)
        self.down2 = DownBlock(base * 2, base * 4)

        self.res1 = ResidualBlock(base * 4)
        self.attn = SelfAttention(base * 4)
        self.res2 = ResidualBlock(base * 4)

        self.up1 = UpBlock(base * 4, base * 2)
        self.up2 = UpBlock(base * 4, base)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        b = self.res2(self.attn(self.res1(x3)))

        u1 = self.up1(b)
        u1 = torch.cat([u1, x2], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        return self.out_conv(u2)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization for stable training."""

    def __init__(self, in_channels: int = 2, base: int = 64):
        super().__init__()

        def disc_block(ch_in: int, ch_out: int, stride: int = 2) -> nn.Sequential:
            return nn.Sequential(
                spectral_norm(nn.Conv2d(ch_in, ch_out, 4, stride=stride, padding=1)),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            disc_block(base, base * 2),
            disc_block(base * 2, base * 4),
            spectral_norm(nn.Conv2d(base * 4, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, y], dim=1))
