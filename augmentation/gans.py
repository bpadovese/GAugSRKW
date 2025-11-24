#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gans.py

Implements GAN-based spectrogram synthesis for data augmentation.
Supports WGAN-GP and DCGAN architectures for generating synthetic
spectrograms of marine mammal vocalizations or other acoustic events.
Includes training, inference-only generation, and optional PCA-based
filtering of generated samples to remove unrealistic outputs.

Author
------
Bruno Padovese (HALLO Project, SFU)
https://github.com/bpadovese
"""

# =============================================================================
# Imports
# =============================================================================
import os
import torch
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image
from torch import nn, optim
from accelerate import Accelerator
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from dev_utils.validate_generation import filter_spectrograms, load_spectrogram

class Generator(nn.Module):
    """Generator network for GAN-based spectrogram synthesis."""
    def __init__(self, z_dim, kernel_size=4, image_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size, 1, 0, bias=False),  # (B, 1024, 4, 4)
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size, 2, 1, bias=False),  # 8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size, 2, 1, bias=False),  # 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size, 2, 1, bias=False),  # 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size, 2, 1, bias=False),  # 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, image_channels, kernel_size, 2, 1, bias=False),  # 128
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

class PhaseShuffle(nn.Module):
    """Applies small random phase shifts to improve discriminator invariance."""
    def __init__(self, shift_range=2):
        super().__init__()
        self.shift_range = shift_range

    def forward(self, x):
        if self.shift_range == 0:
            return x

        phase = int(torch.randint(-self.shift_range, self.shift_range + 1, (1,)))
        return torch.roll(x, shifts=phase, dims=3)  # Shift along width
  
class Discriminator(nn.Module):
    """Discriminator network."""
    def __init__(self, image_channels=1, kernel_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_range=2),

            nn.Conv2d(64, 128, kernel_size, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_range=2),

            nn.Conv2d(128, 256, kernel_size, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_range=2),

            nn.Conv2d(256, 512, kernel_size, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            PhaseShuffle(shift_range=2),

            nn.Conv2d(512, 1, kernel_size, 1, 0),
        )

    def forward(self, x):
        x = self.net(x)
        return x.mean(dim=(2, 3)).squeeze(1)  # Global average pooling

# =============================================================================
# Utilities
# =============================================================================

def gradient_penalty(critic, real, fake, device="cuda"):
    """Compute WGAN-GP gradient penalty."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    prob_interpolated = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)  # Per-sample L2 norm

    # penalty
    gp = ((grad_norm - 1) ** 2).mean()
    return gp, grad_norm.mean()


def save_image_grid(images, path, nrow=4):
    """Save a grid of generated images (denormalized to [0,1])."""
    images = (images + 1) / 2  # denormalize from [-1, 1] to [0, 1]
    utils.save_image(images, path, nrow=nrow)

def load_image_dataset(path, transform=None, limit=None):
    """
    Load an ImageFolder dataset or single-class image folder.

    - If the path contains subfolders, load all classes.
    - If the path is a leaf folder with images, treat it as one class.

    Parameters
    ----------
    path : str
        Directory containing class subfolders or images.
    transform : callable, optional
        Image transform pipeline.
    limit : int, optional
        Limit number of samples for fast experiments.

    Returns
    -------
    dataset : torch.utils.data.Dataset
        Torch dataset object.
    """
    # Check if there are any subdirectories (class folders)
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    if subdirs:
        # Has subdirectories — treat as full ImageFolder
        dataset = ImageFolder(root=path, transform=transform, loader=lambda path: Image.open(path))
        if limit is not None:
            indices = random.sample(range(len(dataset)), min(limit, len(dataset)))
            dataset = Subset(dataset, indices)
    else:
        # Leaf folder with images — treat as single class
        parent_path = os.path.dirname(os.path.normpath(path))
        class_name = os.path.basename(os.path.normpath(path))

        # Load from parent but filter only target class
        full_dataset = ImageFolder(root=parent_path, transform=transform, loader=lambda path: Image.open(path))
        
        # Get the index of the class name
        class_to_idx = full_dataset.class_to_idx
        if class_name not in class_to_idx:
            raise ValueError(f"Class folder '{class_name}' not found in {parent_path}.")
        
        label = class_to_idx[class_name]
        indices = [i for i, (_, l) in enumerate(full_dataset.samples) if l == label]
        if limit is not None:
            indices = random.sample(indices, min(limit, len(indices)))
        
        dataset = Subset(full_dataset, indices)
    
    return dataset


# =============================================================================
# Training Loop
# =============================================================================
def train_loop(config):
    """Main training loop for GAN training."""
    if config['seed'] is not None:
        # Set seeds for reproducibility
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])

    accelerator = Accelerator(mixed_precision=config['mixed_precision'])
    device = accelerator.device

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Models
    G = Generator(config['z_dim'], image_channels=config['channels']).to(device)
    D = Discriminator(config['channels']).to(device)

    # Optimizers
    if config['gan_type'] == "dcgan":
        opt_G = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        opt_D = optim.Adam(D.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    else:
        opt_G = optim.Adam(G.parameters(), lr=config['lr'], betas=(0, 0.9))
        opt_D = optim.Adam(D.parameters(), lr=config['lr'], betas=(0, 0.9))

    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = load_image_dataset(config['dataset'], transform=transform, limit=config['image_limit'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    G, D, opt_G, opt_D, dataloader = accelerator.prepare(G, D, opt_G, opt_D, dataloader)


    # Training loop
    for epoch in range(config['num_epochs']):
        progress_bar = tqdm(
            total=len(dataloader),
            leave=False,  # ✅ Important: Do not keep old bars
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process
        )
        
        # pbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for real, _ in dataloader:
            real = real.to(device)

            # === Train Discriminator ===
            for _ in range(config['critic_iters']):
                noise = torch.randn(real.size(0), config['z_dim'], 1, 1, device=device)
                fake = G(noise).detach()
                
                # Add instance noise
                noise_std = config.get("instance_noise_std", 0.01)
                real_noisy = real + noise_std * torch.randn_like(real)
                fake_noisy = fake + noise_std * torch.randn_like(fake)

                if config['gan_type'] == "wgan-gp":
                    loss_D = -D(real_noisy).mean() + D(fake_noisy).mean()
                    gp, _ = gradient_penalty(D, real_noisy, fake_noisy, device)
                    loss_D_total = loss_D + config['lambda_gp'] * gp
                else:  # DCGAN
                    real_labels = torch.ones(real.size(0), device=device)
                    fake_labels = torch.zeros(real.size(0), device=device)
                    loss_fn = nn.BCEWithLogitsLoss()

                    D_real_out = D(real).squeeze()
                    D_fake_out = D(fake).squeeze()

                    loss_D_real = loss_fn(D_real_out, real_labels)
                    loss_D_fake = loss_fn(D_fake_out, fake_labels)
                    loss_D_total = loss_D_real + loss_D_fake

                opt_D.zero_grad()
                accelerator.backward(loss_D_total)
                opt_D.step()

            # === Train Generator ===
            noise = torch.randn(real.size(0), config['z_dim'], 1, 1, device=device)
            fake = G(noise)
            
            if config['gan_type'] == "wgan-gp":
                loss_G = -D(fake).mean()
            else:
                fake_labels = torch.ones(fake.size(0), device=device) 
                D_fake_out = D(fake).squeeze()
                loss_fn = nn.BCEWithLogitsLoss()
                loss_G = loss_fn(D_fake_out, fake_labels)

            opt_G.zero_grad()
            accelerator.backward(loss_G)
            opt_G.step()

            progress_bar.set_postfix({
                "D": f"{loss_D_total.item():.2f}",
                "G": f"{loss_G.item():.2f}",
            })
            progress_bar.update(1)
        progress_bar.close()

        if accelerator.is_main_process and (epoch + 1) % config['save_image_epochs'] == 0:
            with torch.no_grad():
                noise = torch.randn(real.size(0), config['z_dim'], 1, 1, device=device)
                samples = G(noise).cpu()
                save_image_grid(samples, output_dir / f"epoch_{epoch+1}.png")

    if accelerator.is_main_process:
        torch.save(G.state_dict(), output_dir / "generator_final.pth")
        torch.save(D.state_dict(), output_dir / "critic_final.pth")

# =============================================================================
# Inference / Generation
# =============================================================================

@torch.no_grad()
def generate_only(config):
    """
    Generate synthetic spectrograms using a pretrained generator.
    
    Optionally filters generated samples using PCA-space Mahalanobis
    distance computed from real validation spectrograms.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = Generator(config['z_dim'], image_channels=config['channels']).to(device)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if config['pretrained_generator'] is None:
        raise ValueError("You must provide --pretrained_generator to use --generate_only.")

    # Load weights
    G.load_state_dict(torch.load(config['pretrained_generator'], map_location=device))
    G.eval()

    # Optionally load PCA reference distribution for filtering 
    real_pca_scores = None
    if config["validation_dir"]:
        real_spectrogram_files = [
            os.path.join(config["validation_dir"], f) for f in os.listdir(config["validation_dir"]) if f.endswith('.png')
        ]
        
        real_specs = np.array([
            load_spectrogram(f, image_shape=(config['image_size'], config['image_size'])).flatten()
            for f in real_spectrogram_files
        ])

        pca = PCA(n_components=2)
        real_pca_scores = pca.fit_transform(real_specs)

    total = 0
    pbar = tqdm(total=config["num_samples"], desc="Generating", ncols=100)

    while total < config["num_samples"]:
        # Generate noise and samples with 1 channel
        noise = torch.randn(config['batch_size'], config['z_dim'], 1, 1, device=device)
        samples = G(noise).cpu()
        # denormalizing and transforming to PIL
        imgs = ((samples + 1) / 2 * 255).clamp(0, 255).byte()
        
        # Optional PCA filtering
        if real_pca_scores is not None:
            imgs_np = imgs.cpu().numpy()
            _, keep = filter_spectrograms(imgs_np, real_pca_scores, pca)
            imgs = [img for i, img in enumerate(imgs) if keep[i]]
        
        for img in imgs:
            if total >= config["num_samples"]:
                break
            img_pil = to_pil_image(img)
            img_pil.save(output_dir / f"gan_sample_{total}.png")
            total += 1
            pbar.update(1)

    print(f"Saved {config['num_samples']} GAN samples to {config['output_dir']}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to image dataset. Required unless --generate_only is set.")
    parser.add_argument("--generate_only", action="store_true",
                    help="Only generate samples using a pretrained generator (no training).")
    parser.add_argument("--pretrained_generator", type=str, default=None,
                        help="Path to the pretrained generator .pth file")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of samples to generate when in generate_only mode.")
    parser.add_argument("--output_dir", type=str, default="wgan_output")
    parser.add_argument("--image_limit", type=int, default=None, help="Limit the number of training images loaded. Useful for fast experiments.")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--gan_type", type=str, choices=["wgan-gp", "dcgan"], default="dcgan",
                    help="Which GAN variant to train.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--validation_dir", type=str, default=None, help="Directory with real spectrogram PNGs")
    parser.add_argument("--critic_iters", type=int, default=5)
    parser.add_argument("--lambda_gp", type=float, default=10.0)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16"])
    parser.add_argument("--save_image_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    if not args.generate_only and args.dataset is None:
        raise ValueError("You must provide a dataset path unless --generate_only is used.")
    
    config = vars(args)
    if config["generate_only"]:
        generate_only(config)
    else:
        train_loop(config)