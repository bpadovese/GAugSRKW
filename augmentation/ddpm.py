#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ddpm.py

Denoising Diffusion Probabilistic Model (DDPM) for spectrogram generation.
Trains a DDPM using a UNet backbone on 128x128 (or other size) spectrogram
images. Supports loading pretrained DDPM models, logging via Accelerate,
and periodic generation of sample images during training.

Author
------
Bruno Padovese (HALLO Project, SFU)
https://github.com/bpadovese
"""

# =============================================================================
# Imports
# =============================================================================


import os
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from accelerate import Accelerator
from diffusers import DDPMScheduler, DDPMPipeline, DiffusionPipeline, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from PIL import Image
from dev_utils.nn import unet
from dev_utils.validate_generation import filter_spectrograms, load_spectrogram


# =============================================================================
# Utility Functions
# =============================================================================

def load_image_dataset(path, transform=None):
    """
    Load image dataset automatically (single folder = one class,
    subfolders = ImageFolder classes).

    Parameters
    ----------
    path : str
        Path to root dataset folder.
    transform : callable
        Transform pipeline for images.

    Returns
    -------
    dataset : torch.utils.data.Dataset
    """
    # Check if there are any subdirectories (class folders)
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    if subdirs:
        # Standard ImageFolder structure: root/class_x/*.png
        return ImageFolder(root=path, transform=transform, loader=lambda path: Image.open(path))
    else:
        # Leaf directory: treat folder name as class
        parent = Path(path).parent
        class_name = Path(path).name

        # Load from parent but filter only target class
        full_dataset = ImageFolder(
            root=str(parent),
            transform=transform,
            loader=lambda p: Image.open(p)
        )
        
        # Get the index of the class name
        class_to_idx = full_dataset.class_to_idx
        if class_name not in class_to_idx:
            raise ValueError(f"Class folder '{class_name}' not found in {parent}.")
        
        label_idx = class_to_idx[class_name]
        indices = [i for i, (_, l) in enumerate(full_dataset.samples) if l == label_idx]
    
    return Subset(full_dataset, indices)


# =============================================================================
# Evaluation (sampling images)
# =============================================================================

def evaluate(config, epoch, pipeline, generator):
    """
    Generate sample images using the current DDPM pipeline.

    Parameters
    ----------
    config : dict
        Training configuration.
    epoch : int
        Epoch number (for filename indexing).
    pipeline : DDPMPipeline
        Diffusers pipeline with UNet + scheduler.
    generator : torch.Generator
        RNG for deterministic sampling.
    """
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    test_dir = Path(config['output_dir']) / "samples"
    test_dir.mkdir(parents=True, exist_ok=True)

    images = pipeline(
        batch_size=config['eval_batch_size'],
        generator=generator,
        num_inference_steps=pipeline.scheduler.config.num_train_timesteps,
        return_dict=False,
    )[0]

    for idx, image in enumerate(images):  # `images` is a list of PIL.Image objects
        image.save(os.path.join(test_dir, f"image_epoch{epoch}_sample{idx}.png"))


# =============================================================================
# Training Loop
# =============================================================================

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    """
    Full DDPM training loop using HuggingFace Accelerate.

    Each training step:
    - Sample random timestep t
    - Add noise according to scheduler q(x_t | x_0)
    - Predict noise via UNet
    - Optimize MSE(noise_pred, true_noise)
    """
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="tensorboard",
        project_dir=Path(config['output_dir']) / "logs",
    )

    if accelerator.is_main_process:
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
        accelerator.init_trackers("ddpm_training")

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # train the model
    for epoch in range(config['num_epochs']):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}"
        )

        for _, (clean_images, _) in enumerate(train_dataloader):
            
            # Sample Gaussian noise
            noise = torch.randn_like(clean_images)

            # Sample random timesteps for each image
            timesteps = torch.randint(
                0, noise_scheduler.config['num_train_timesteps'], (clean_images.shape[0],), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise added at step t
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                # MSE loss between predicted noise and true noise
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging 
            global_step += 1
            
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            progress_bar.update(1)

        progress_bar.close()
        accelerator.wait_for_everyone()

        # After each epoch optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # Conditions
            save_samples = ((epoch + 1) % config['save_image_epochs'] == 0)
            save_model = ((epoch + 1) % config['save_model_epochs'] == 0)
            last_epoch = (epoch == config['num_epochs'] - 1)

            if save_samples or save_model or last_epoch:
                print("saving image...")
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler
                )          
                
            if save_model or last_epoch:
                pipeline.save_pretrained(config['output_dir'])

            if save_samples:
                generator = torch.Generator(device=clean_images.device).manual_seed(42)
                evaluate(config, epoch, pipeline, generator)
            
        accelerator.wait_for_everyone()

    accelerator.end_training()


# =============================================================================
# DDPM Inference (Generate Only)
# =============================================================================

@torch.no_grad()
def ddpm_generate(config):
    """
    Generate spectrogram images using a pretrained DDPM model.
    Supports optional PCA-based realism filtering, consistent
    with GAN and VAE generation scripts.

    Parameters
    ----------
    config : dict
        Fields include:
        - pretrained_model : str
        - num_samples : int
        - output_dir : str
        - batch_size : int
        - num_timesteps : int
        - validation : str or None
        - seed : int or None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.pretrained_model is None:
        raise ValueError(
            "You must provide --pretrained_model when using --generate_only."
        )

    # Set seeds for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    # Load pretrained model
    pipeline = DiffusionPipeline.from_pretrained(config.pretrained_model).to(device)

    # Swap scheduler for DDIM (faster)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True, leave=True, desc="Pipeline Progress") # for some reason, leave True not working

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    # Load validation data (optional)
    real_pca_scores = None
    if config.validation_dir:
        real_files = [
            os.path.join(config.validation_dir, f) for f in os.listdir(config.validation_dir) if f.endswith(".png")
        ]

        real_specs_flat = np.array([
            load_spectrogram(f, image_shape=(config.image_size, config.image_size)).flatten() for f in real_files
        ])

        pca = PCA(n_components=2)
        real_pca_scores = pca.fit_transform(real_specs_flat)


    # Generation loop
    num_saved = 0
    batch_index = 0

    with tqdm(total=config.num_samples, desc="Generating DDPM samples") as pbar:

        while num_saved < config.num_samples:
            batch_index += 1
            batch_remaining = config.num_samples - num_saved
            batch_size = min(config.batch_size, batch_remaining)

            # Generate batch with DDIM/ DDPM
            images = pipeline(
                batch_size=batch_size,
                generator=None,
                num_inference_steps=config.num_timesteps,
                return_dict=False
            )[0]

            # OPTIONAL PCA FILTER
            if real_pca_scores is not None:
                imgs_np = np.array([np.array(img) for img in images]) # Convert no np array
                _, keep = filter_spectrograms(imgs_np, real_pca_scores, pca)
                images = [img for i, img in enumerate(images) if keep[i]]

            # Save imgs as png
            for img in images:
                if num_saved >= config.num_samples:
                    break
                img.save(output_dir / f"diffusion_{num_saved}.png")
                num_saved += 1
                pbar.update(1)

    print(f"\nSaved {num_saved} DDPM samples to {output_dir}")


# =============================================================================
# Main Function
# =============================================================================

def main(dataset,
        pretrained_model=None,
        image_size=128,
        train_batch_size=8,
        eval_batch_size=8,
        num_epochs=50,
        num_timesteps=1000,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_warmup_steps=500,
        save_image_epochs=10,
        save_model_epochs=10,
        mixed_precision="no",
        output_dir="ddpm_output",
        overwrite_output_dir=True,
        seed=None
        ):

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Load images from the folder
    train_dataset = load_image_dataset(dataset, transform=train_transform)
    print(f"Loaded dataset with {len(train_dataset)} images.")

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    
    if pretrained_model:
        image_pipe = DDPMPipeline.from_pretrained(pretrained_model)
        model = image_pipe.unet
        noise_scheduler = image_pipe.scheduler
    else:
        model = unet(sample_size=image_size, channels=1)
        noise_scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    from accelerate import notebook_launcher

    #  Create a config dictionary from the parsed arguments
    config = {
        "dataset": dataset,
        "eval_batch_size": eval_batch_size,
        "num_epochs": num_epochs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "save_image_epochs": save_image_epochs,
        "save_model_epochs": save_model_epochs,
        "mixed_precision": mixed_precision,
        "output_dir": output_dir,
        "overwrite_output_dir": overwrite_output_dir,
        "seed": seed
    }
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    # notebook_launcher(train_loop, args, num_processes=1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training configuration.")
    parser.add_argument("--dataset", type=str, help="Path to the image folder.")
    parser.add_argument("--generate_only", action="store_true",
                    help="Generate samples using a pretrained DDPM (no training).")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to a pretrained DDPM model")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate (only with --generate_only).")
    parser.add_argument("--image_size", type=int, default=128, help="Generated image resolution.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of timesteps for the scheduler.")
    parser.add_argument("--validation_dir", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--save_image_epochs", type=int, default=10, help="Epochs interval to save generated images.")
    parser.add_argument("--save_model_epochs", type=int, default=30, help="Epochs interval to save the model.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16"], help="Mixed precision setting.")
    parser.add_argument("--output_dir", type=str, default="ddpm-butterflies-128", help="Output directory for the model.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory if it exists.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    
    args = parser.parse_args()
    
    if args.generate_only:
        ddpm_generate(args)
    else:
        main(
            args.dataset,
            pretrained_model=args.pretrained_model,
            image_size=args.image_size,
            train_batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            num_epochs=args.num_epochs,
            num_timesteps=args.num_timesteps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_warmup_steps=args.lr_warmup_steps,
            save_image_epochs=args.save_image_epochs,
            save_model_epochs=args.save_model_epochs,
            mixed_precision=args.mixed_precision,
            output_dir=args.output_dir,
            overwrite_output_dir=args.overwrite_output_dir,
            seed=args.seed,
        )
    