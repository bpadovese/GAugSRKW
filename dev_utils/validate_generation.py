#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_generation.py

Spectrogram filtering utilities based on PCA distance metrics.
Provides tools to reject outlier synthetic spectrograms that differ
substantially from real data distributions using Mahalanobis distance
in PCA space.

Author
------
Bruno Padovese (HALLO Project, SFU)
https://github.com/bpadovese
"""

import numpy as np
from PIL import Image
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

def load_spectrogram(image_path, image_shape):
    """
    Load a spectrogram image and convert it to a grayscale NumPy array.

    Parameters
    ----------
    image_path : str or Path
        Path to the spectrogram image.
    image_shape : tuple(int, int)
        Target size (width, height) for resizing the image.

    Returns
    -------
    np.ndarray
        Grayscale spectrogram image as a 2D NumPy array.
    """
    img = Image.open(image_path).resize(image_shape)
    return np.array(img)

def filter_spectrograms(spectrograms, real_pca_scores, pca, percentile=95):
    """
    Filter generated spectrograms using PCA-space Mahalanobis distance.

    This function compares the PCA-projected distribution of generated
    spectrograms against that of real spectrograms, retaining only those
    samples that fall within a specified percentile of Mahalanobis distance
    relative to the real data cluster.

    Parameters
    ----------
    spectrograms : np.ndarray
        Array of generated spectrograms with shape (N, H, W).
    real_pca_scores : np.ndarray
        PCA scores of real spectrograms (used as reference distribution).
    pca : sklearn.decomposition.PCA
        Trained PCA model fitted on real spectrograms.
    percentile : float, optional
        Percentile cutoff (default: 95). Generated samples with distances
        above this threshold are rejected.

    Returns
    -------
    filtered_spectrograms : np.ndarray
        Array of spectrograms that pass the Mahalanobis filter.
    keep_indices : np.ndarray (bool)
        Boolean mask indicating which spectrograms were retained.
    """
    # Flatten generated spectrograms and project them into PCA space
    gen_pca_scores = pca.transform(spectrograms.reshape(len(spectrograms), -1))

    # Compute Mahalanobis distance of generated samples to the real PCA cluster
    mean_real = np.mean(real_pca_scores, axis=0)
    cov_real = np.cov(real_pca_scores, rowvar=False)
    cov_inv = inv(cov_real)

    distances = [
        mahalanobis(score, mean_real, cov_inv)
        for score in gen_pca_scores
    ]

    # Define distance threshold based on real sample distribution
    real_distances = [
        mahalanobis(score, mean_real, cov_inv)
        for score in real_pca_scores
    ]

    # Define a threshold for filtering (e.g., 95% of real spectrograms)
    threshold = np.percentile(real_distances, percentile)

    # Keep spectrograms within threshold
    keep_indices = distances <= threshold
    filtered_spectrograms = spectrograms[keep_indices]
    
    return filtered_spectrograms, keep_indices