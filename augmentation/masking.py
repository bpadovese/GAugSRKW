#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask Construction and Background Merging
--------------------------------------------------------------------------

Implements the exact pipeline described in the paper:
1. Load spectrogram
2. PCA-based background estimation (PC1 reconstruction)
3. Subtract reconstructed background from original
4. Percentile-based thresholding
5. Save final sparse vocalization mask
6. Merge the generated masks with background spectrogram images using an
    additive overlay

Author: Bruno Padovese (HALLO Project, SFU)
"""

import os
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt


# --------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------

def load_spectrogram(path):
    """
    Load a spectrogram PNG as a 2D float32 array.

    Parameters
    ----------
    path : str
        Path to the PNG spectrogram file.

    Returns
    -------
    np.ndarray (H, W)
        Grayscale spectrogram matrix in float32 format.
    """
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def save_image(array, path, cmap="viridis", colorbar=False):
    """
    Save a spectrogram-like array to disk.

    Parameters
    ----------
    array : np.ndarray
        2D matrix representing a spectrogram or mask.

    path : str
        Output file path.

    cmap : str
        Matplotlib colormap used when colorbar=True.

    colorbar : bool
        If True, saves using matplotlib with colorbar and axes.
        If False, saves a raw uint8 image without axes or colorbar.
    """
    if colorbar:
        plt.figure()
        plt.imshow(array, aspect="auto", origin="lower", cmap=cmap)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        
        # No axes, no frame, no colorbar
        # plt.imshow(array, cmap=cmap, aspect="auto", origin="lower")
        # plt.axis("off")
        # plt.margins(0, 0)
        # plt.gca().set_frame_on(False)
        # plt.savefig(path, bbox_inches="tight", pad_inches=0)
        # plt.close()
    else:
        img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        img.save(path)


def pca_background_subtraction(spec, n_components=1):
    """
    Perform PCA-based background estimation on a spectrogram.

    The spectrogram is decomposed via PCA, reconstructed using only
    the dominant component(s), and the background estimate is subtracted
    from the original to produce a residual emphasizing vocalizations.

    Parameters
    ----------
    spec : np.ndarray (H, W)
        Input spectrogram.

    n_components : int
        Number of PCA components to retain. Typically 1.

    Returns
    -------
    recon : np.ndarray
        PCA reconstruction using the first component(s).

    residual : np.ndarray
        Non-negative difference between the original spectrogram
        and its PCA reconstruction.
    """

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(spec)
    recon = pca.inverse_transform(scores)

    residual = np.clip(spec - recon, 0, None)

    return recon, residual


def percentile_threshold(img, percentile=95):
    """
    Apply percentile-based thresholding to a residual spectrogram.

    All values below the chosen percentile are set to zero. This creates a
    sparse mask highlighting high-intensity vocalization pixels.

    Parameters
    ----------
    img : np.ndarray
        Residual spectrogram or similar matrix.

    percentile : float
        Percentile threshold (0-100). Typical values: 90-99.

    Returns
    -------
    np.ndarray
        Thresholded mask where values below the percentile are zeroed.
    """
    th = np.percentile(img, percentile)
    mask = np.where(img >= th, img, 0)
    return mask


# --------------------------------------------------------------
# Mask construction
# --------------------------------------------------------------

def create_masks(
    input_folder,
    output_folder,
    percentile=95,
    n_components=1,
    sample_size=None,
    seed=None,
    save_intermediate=False,
    colorbar=False
):
    """
    Build sparse vocalization masks using PCA background subtraction
    followed by percentile-based thresholding.

    This pipeline implements the processing described in the accompanying
    paper: PCA background estimation, subtraction, thresholding, and mask
    generation. Optionally saves intermediate PCA reconstructions and residuals.

    Parameters
    ----------
    input_folder : str
        Folder containing spectrogram PNGs.

    output_folder : str
        Root directory for all generated outputs.

    percentile : float
        Percentile threshold for sparsifying the residual.

    n_components : int
        Number of PCA components used for background reconstruction.

    sample_size : int or None
        If set, randomly selects a subset of files to process.

    seed : int or None
        Random seed for reproducible sampling.

    save_intermediate : bool
        If True, saves original spectrograms, PCA reconstructions,
        and residuals in dedicated subdirectories.

    colorbar : bool
        If True, saves images using matplotlib with colorbars.

    Returns
    -------
    None
        Saves masks (and optional intermediates) to disk.
    """

    os.makedirs(output_folder, exist_ok=True)

    mask_dir = os.path.join(output_folder, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    if save_intermediate:
        orig_dir = os.path.join(output_folder, "original")
        modes_dir = os.path.join(output_folder, "modes")
        residual_dir = os.path.join(output_folder, "residual")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(modes_dir, exist_ok=True)
        os.makedirs(residual_dir, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    if seed is not None:
        random.seed(seed)

    if sample_size is not None:
        files = random.sample(files, min(sample_size, len(files)))

    for fname in tqdm(files, desc="Building masks"):
        path = os.path.join(input_folder, fname)
        spec = load_spectrogram(path)

        # Step 1: PCA background subtraction
        recon, residual = pca_background_subtraction(spec, n_components)

        # Step 2: Percentile thresholding
        mask = percentile_threshold(residual, percentile)

        # Save final mask
        save_image(mask, os.path.join(mask_dir, fname), colorbar=colorbar)

        # Save intermediate outputs
        if save_intermediate:
            save_image(spec, os.path.join(orig_dir, fname), colorbar=colorbar)
            save_image(recon, os.path.join(modes_dir, fname), colorbar=colorbar)
            save_image(residual, os.path.join(residual_dir, fname), colorbar=colorbar)

    print(f"Done. Masks saved to {mask_dir}")

# =============================================================================
# Mask + Background Merging
# =============================================================================

def merge_spectrograms(
    mask_folder,
    background_folder,
    output_folder,
    num_samples=5,
    seed=None,
    intensity=1.0,
    cmap=None,
    colorbar=False,
    save_original_backgrounds=False
):
    """
    Merge sparse masks with background spectrogram images to generate
    synthetic spectrogram examples.

    Parameters
    ----------
    mask_folder : str
        Folder containing sparse mask PNGs.
    background_folder : str
        Folder of background spectrogram PNGs.
    output_folder : str
        Destination folder for merged spectrograms.
    num_samples : int
        Number of backgrounds sampled per mask.
    seed : int or None
        Random seed for reproducible sampling.
    intensity : float
        Scaling factor applied to mask before merging.
    cmap : str or None
        Matplotlib colormap (only used when colorbar=True).
    colorbar : bool
        If True, save merged images using matplotlib.
    save_original_backgrounds : bool
        If True, saves copies of the chosen backgrounds.
    """
    os.makedirs(output_folder, exist_ok=True)

    masks = [f for f in os.listdir(mask_folder) if f.endswith(".png")]
    backgrounds = [f for f in os.listdir(background_folder) if f.endswith(".png")]

    if len(masks) == 0:
        raise ValueError("No masks found in mask_folder.")
    if len(backgrounds) == 0:
        raise ValueError("No backgrounds found in background_folder.")
    
    if seed is not None:
        random.seed(seed)

    if save_original_backgrounds:
        bg_save = os.path.join(output_folder, "saved_backgrounds")
        os.makedirs(bg_save, exist_ok=True)
    
    total_generated = 0
    pbar = tqdm(total=num_samples, desc="Merging masks with backgrounds")
    
    # Keep merging until total count reached
    while total_generated < num_samples:
        
        # Randomly pick mask & background
        mask_file = random.choice(masks)
        bg_file = random.choice(backgrounds)

        mask = load_spectrogram(os.path.join(mask_folder, mask_file)).astype(np.int32)
        background = load_spectrogram(os.path.join(background_folder, bg_file)).astype(np.int32)

        if mask.shape != background.shape:
            raise ValueError(f"Shape mismatch: {mask_file} vs {bg_file}")

        if save_original_backgrounds:
            save_image(background, os.path.join(bg_save, bg_file),
                       colorbar=colorbar, cmap=cmap)

        # Additive merge
        merged = np.clip(mask * intensity + background, 0, 255).astype(np.uint8)

        out_name = f"merged_{total_generated}_{Path(mask_file).stem}.png"
        out_path = os.path.join(output_folder, out_name)
        save_image(merged, out_path, colorbar=colorbar, cmap=cmap)
        
        total_generated += 1
        pbar.update(1)
    
    pbar.close()
    print(f"Saved {num_samples} merged examples to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Create vocalization masks via PCA background subtraction.")
    parser.add_argument("--mode", choices=["mask", "merge"], required=True)

    # Mask creation
    parser.add_argument("--input_folder", type=str, help="Folder of input spectrograms.")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Percentile threshold for mask sparsification.")
    parser.add_argument("--n_components", type=int, default=1,
                        help="Number of PCA components (typically 1).")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Optionally process only a sample of files.")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save PCA reconstructions and residuals.")
    # Merge Mode
    parser.add_argument("--mask_folder", type=str)
    parser.add_argument("--background_folder", type=str)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--save_original_backgrounds", action="store_true")

    #Shared
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to store the results.")
    parser.add_argument("--cmap", type=str, default="gray")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling.")
    parser.add_argument("--colorbar", action="store_true",
                        help="Save images with matplotlib + colorbar.")

    args = parser.parse_args()
    
    if args.mode == "mask":
        create_masks(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            percentile=args.percentile,
            n_components=args.n_components,
            sample_size=args.sample_size,
            seed=args.seed,
            save_intermediate=args.save_intermediate,
            colorbar=args.colorbar,
        )
    elif args.mode == "merge":
        merge_spectrograms(
            mask_folder=args.mask_folder,
            background_folder=args.background_folder,
            output_folder=args.output_folder,
            num_samples=args.num_samples,
            seed=args.seed,
            intensity=args.intensity,
            cmap=args.cmap,
            colorbar=args.colorbar,
            save_original_backgrounds=args.save_original_backgrounds,
        )

if __name__ == "__main__":
    main()
