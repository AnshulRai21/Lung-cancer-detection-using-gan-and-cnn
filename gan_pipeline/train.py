"""Training script for advanced cGAN-based lung CT enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from gan_pipeline.architectures import AttentionResUNetGenerator, PatchDiscriminator
from gan_pipeline.dataset import LungCTPairDataset
from gan_pipeline.losses import MultiObjectiveLoss
from gan_pipeline.metrics import compute_psnr_ssim


def postprocess_enhancement(image_tensor: torch.Tensor) -> torch.Tensor:
    """Apply CLAHE + normalization for visible quality boost."""
    image = image_tensor.detach().cpu().squeeze().numpy()
    image = ((image + 1.0) * 127.5).clip(0, 255).astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    enhanced = enhanced.astype('float32') / 127.5 - 1.0
    return torch.from_numpy(enhanced).unsqueeze(0).unsqueeze(0)


def save_epoch_visuals(x: torch.Tensor, fake: torch.Tensor, out_dir: Path, epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    x_np = ((x[0, 0].detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype('uint8')
    f_np = ((fake[0, 0].detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype('uint8')

    cv2.imwrite(str(out_dir / f'epoch_{epoch:04d}_input.png'), x_np)
    cv2.imwrite(str(out_dir / f'epoch_{epoch:04d}_generated.png'), f_np)


def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = LungCTPairDataset(args.data_dir, image_size=args.image_size, augment=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    net_g = AttentionResUNetGenerator().to(device)
    net_d = PatchDiscriminator().to(device)

    opt_g = Adam(net_g.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = Adam(net_d.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion = MultiObjectiveLoss(lambda_l1=args.lambda_l1, lambda_perceptual=args.lambda_perceptual).to(device)

    for epoch in range(1, args.epochs + 1):
        net_g.train()
        net_d.train()
        running_g = 0.0
        running_d = 0.0

        prog = tqdm(dl, desc=f'Epoch {epoch}/{args.epochs}')
        for x, y in prog:
            x = x.to(device)
            y = y.to(device)

            # ---- Train Discriminator ----
            with torch.no_grad():
                fake_for_d = net_g(x)
            d_real = net_d(x, y)
            d_fake = net_d(x, fake_for_d)

            loss_d_real = criterion.adversarial_loss(d_real, is_real=True, smooth=0.1)
            loss_d_fake = criterion.adversarial_loss(d_fake, is_real=False, smooth=0.0)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- Train Generator ----
            fake = net_g(x)
            fake = postprocess_enhancement(fake)
            fake = fake.to(device)

            d_fake_for_g = net_d(x, fake)
            g_losses = criterion.generator_loss(d_fake_for_g, fake, y)

            opt_g.zero_grad(set_to_none=True)
            g_losses['total'].backward()
            opt_g.step()

            running_d += loss_d.item()
            running_g += g_losses['total'].item()
            prog.set_postfix(loss_d=loss_d.item(), loss_g=g_losses['total'].item())

        # Save one visual sample per epoch for demo proof
        save_epoch_visuals(x, fake, Path(args.visual_dir), epoch)

        # Quick epoch metrics on last batch sample
        psnr, ssim = compute_psnr_ssim(
            fake[0, 0].detach().cpu().numpy(),
            y[0, 0].detach().cpu().numpy(),
        )

        print(
            f'Epoch {epoch}: '
            f'G={running_g/len(dl):.4f} '
            f'D={running_d/len(dl):.4f} '
            f'PSNR={psnr:.2f} SSIM={ssim:.4f}'
        )

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.save(net_g.state_dict(), str(Path(args.checkpoint_dir) / 'advanced_gan_generator.pth'))
    torch.save(net_d.state_dict(), str(Path(args.checkpoint_dir) / 'advanced_gan_discriminator.pth'))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train advanced cGAN for lung CT enhancement.')
    parser.add_argument('--data-dir', type=str, required=True, help='Root dataset dir with input/ and target/ subfolders.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda-l1', type=float, default=100.0)
    parser.add_argument('--lambda-perceptual', type=float, default=10.0)
    parser.add_argument('--visual-dir', type=str, default='training_visuals')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    train(args)
