#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vae.py

Variational Autoencoder (VAE) for spectrogram generation and data augmentation.
Trains a VAE on spectrogram images and optionally generates
synthetic samples from the learned latent space. Supports PCA-based filtering
of generated samples to remove unrealistic outputs.

Author
------
Bruno Padovese (HALLO Project, SFU)
https://github.com/bpadovese
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from torch import nn, optim
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image
from accelerate import Accelerator
from sklearn.decomposition import PCA

from dev_utils.validate_generation import filter_spectrograms, load_spectrogram

# =============================================================================
# Model Definition
# =============================================================================

class VAE(nn.Module):
    def __init__(self, z_dim=20, image_channels=1):
        super().__init__()
        # Encoder: Conv -> ReLU stack -> flatten
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Latent space: mean and log-variance heads
        self.fc_mu = nn.Linear(128 * 16 * 16, z_dim)
        self.fc_logvar = nn.Linear(128 * 16 * 16, z_dim)

        # Decoder: fully connected -> unflatten -> ConvTranspose stack
        self.fc_decode = nn.Linear(z_dim, 128 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, 2, 1),
            nn.Tanh(), # Output [-1, 1]
        )

    def encode(self, x):
        """Encode input spectrogram into latent mean and log-variance."""
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to allow backprop through sampling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent variable z back into a spectrogram image."""
        x = self.fc_decode(z)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# =============================================================================
# Loss Function
# =============================================================================

def vae_loss(recon_x, x, mu, logvar, beta=1.0, loss_type='mse'):
    """
    Compute the VAE loss (reconstruction + KL divergence).

    Parameters
    ----------
    recon_x : torch.Tensor
        Reconstructed image.
    x : torch.Tensor
        Original input image.
    mu : torch.Tensor
        Latent mean vector.
    logvar : torch.Tensor
        Latent log-variance vector.
    beta : float
        Weight of the KL divergence term (β-VAE).
    loss_type : str
        Reconstruction loss type: 'mse' or 'bce'.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    if loss_type == 'bce':
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence term
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld

# =============================================================================
# Utility Functions
# =============================================================================

def save_image_grid(images, path, nrow=4):
    """Save a grid of reconstructed or generated images."""
    images = (images + 1) / 2 # Denormalize from [-1, 1] to [0, 1]
    utils.save_image(images, str(path), nrow=nrow)

def load_image_dataset(path, transform=None, limit=None):
    """
    Load image dataset automatically from ImageFolder or single folder.
    Supports both class-subfolder and single-class directory structures.
    """
    path = Path(path)
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if subdirs:
        dataset = ImageFolder(root=str(path), transform=transform, loader=lambda p: Image.open(p))
        if limit is not None:
            indices = random.sample(range(len(dataset)), min(limit, len(dataset)))
            dataset = Subset(dataset, indices)
    else:
        parent_path = path.parent
        class_name = path.name
        full_dataset = ImageFolder(root=str(parent_path), transform=transform, loader=lambda p: Image.open(p))
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
# Training
# =============================================================================

def train_loop(config):
    accelerator = Accelerator(mixed_precision=config['mixed_precision'])
    device = accelerator.device

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = load_image_dataset(config['dataset'], transform=transform, limit=config['image_limit'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    model = VAE(config['z_dim'], image_channels=config['channels']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # === Training Loop ===
    for epoch in range(config['num_epochs']):
        model.train()
        with tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{config['num_epochs']}", 
              leave=False, disable=not accelerator.is_main_process) as pbar:
            for x, _ in pbar:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar, beta=config['beta'], loss_type=config['loss_type'])
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                # Update tqdm bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save sample reconstructions every few epochs
        if accelerator.is_main_process and (epoch + 1) % config['save_image_epochs'] == 0:
            model.eval()
            with torch.no_grad():
                noise = torch.randn(16, config['z_dim'], device=device)
                samples = model.decode(noise).cpu()
                save_image_grid(samples, f"{config['output_dir']}/epoch_{epoch+1}.png")

    # Save final model
    if accelerator.is_main_process:
        torch.save(model.state_dict(), os.path.join(config['output_dir'], "vae_final.pth"))

# =============================================================================
# Inference / Generation
# =============================================================================

@torch.no_grad()
def generate_only(config):
    """
    Generate synthetic spectrograms from a pretrained VAE.

    Optionally filters samples based on PCA-space Mahalanobis
    distance from real spectrograms.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE(config['z_dim'], image_channels=config['channels']).to(device)
    model.load_state_dict(torch.load(config['pretrained_vae'], map_location=device))
    model.eval()

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation spectrograms (if any)
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
        z = torch.randn(config['batch_size'], config['z_dim'], device=device)
        samples = model.decode(z).cpu()
        imgs = ((samples + 1) / 2 * 255).clamp(0, 255).byte()

        if real_pca_scores is not None:
            imgs_np = imgs.numpy()
            _, keep = filter_spectrograms(imgs_np, real_pca_scores, pca)
            imgs = [img for i, img in enumerate(imgs) if keep[i]]

        for img in imgs:
            if total >= config["num_samples"]:
                break
            img_pil = to_pil_image(img)
            img_pil.save(output_dir / f"vae_sample_{total}.png")
            total += 1

            # Convert to numpy and normalize to [0, 1]
            spec = img.squeeze().numpy().astype(np.float32)
            spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)  # Normalize
            total += 1
            pbar.update(1)

    print(f"Saved {config['num_samples']} VAE samples to {config['output_dir']}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to image dataset.")
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--pretrained_vae", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="vae_output")
    parser.add_argument("--image_limit", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--validation_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16"])
    parser.add_argument("--save_image_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--beta", type=float, default=1.0, help="Beta weight for KL loss.")
    parser.add_argument("--loss_type", type=str, choices=["mse", "bce"], default="mse")

    args = parser.parse_args()
    config = vars(args)

    if config["generate_only"]:
        generate_only(config)
    else:
        train_loop(config)
