import torch
import numpy as np
from noise import pnoise2, pnoise3
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_fill_holes, label

import os
import numpy as np
import random
from abc import abstractmethod

from ..configuration.mask import MaskConfig, PresampledMaskConfig

class MaskSampler:
    def __init__(self, mask_config: MaskConfig):
        self.config = mask_config

    @abstractmethod
    def sample(self, index):
        """
        Samples a mask from the directory using a consistent random mapping.

        Parameters:
            index (int): The index for sampling.

        Returns:
            numpy.ndarray: The sampled mask.
        """
        raise NotImplementedError
    
class DirectorySampler(MaskSampler):

    def __init__(self, mask_config: PresampledMaskConfig):
        super().__init__(mask_config)
        self.masks = self._load_masks()
    
    def _load_masks(self):
        """
        Loads all .npy files in the directory and returns their file paths.

        Returns:
            list: List of .npy file paths in the directory.
        """
        return [os.path.join(self.config.directory, f) for f in os.listdir(self.config.directory) if f.endswith('.npy')]
    
    def sample(self, index):
        if not self.masks:
            raise ValueError("No .npy files found in the directory.")

        # Use a consistent random mapping based on the index
        random.seed(index)
        selected_file = random.choice(self.masks)

        # Load and return the mask
        return np.load(selected_file)

def unique_seed(img_idx, mask_idx, total_masks):
    return img_idx * total_masks + mask_idx

def generate_perlin_mask_with_contour_smoothing(image_size, scale=100, octaves=4, persistence=0.5, lacunarity=2.0, threshold=0.0, sigma=2.0, seed=123):
    """
    Generate a binary mask with smooth contours using Perlin noise.

    Parameters:
        image_size (tuple): Size of the image (2D or 3D).
        scale (float): Scale of the noise (higher values zoom out the noise pattern).
        octaves (int): Number of noise layers blended together for detail.
        persistence (float): Controls amplitude of octaves (higher = more detail).
        lacunarity (float): Controls frequency of octaves (higher = more detail).
        threshold (float): Threshold to create a binary mask.
        sigma (float): Standard deviation for Gaussian smoothing.
        seed (int): seed controlling perlin noise generation, for deterministically generate it. Set to image index.

    Returns:
        np.ndarray: Binary mask (0 or 1) with smooth contours, size (per_sample, height, width)
    """
    if len(image_size) == 2:  # 2D case (height, width)
        height, width = image_size
        depth = 1  # Single layer for 2D
        noise_grid = np.zeros((height, width))
        
        # Generate 2D Perlin noise
        for y in range(height):
            for x in range(width):
                noise_value = pnoise2(x / scale, 
                                      y / scale, 
                                      octaves=octaves, 
                                      persistence=persistence, 
                                      lacunarity=lacunarity, 
                                      repeatx=width, 
                                      repeaty=height, 
                                      base=seed)  # Use seed for reproducibility
                noise_grid[y, x] = noise_value
                
        # Normalize and smooth the noise
        noise_grid = (noise_grid - noise_grid.min()) / (noise_grid.max() - noise_grid.min())
        smoothed_noise = gaussian_filter(noise_grid, sigma=sigma)
        
        # Apply threshold to create a binary mask
        binary_mask = (smoothed_noise > threshold).astype(np.uint8)
        return binary_mask
    
    elif len(image_size) == 3:  # 3D case (height, width, depth)
        height, width, depth = image_size
        noise_grid = np.zeros((height, width, depth))
        
        # Generate 3D Perlin noise
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    noise_value = pnoise3(x / scale, 
                                          y / scale, 
                                          z / scale, 
                                          octaves=octaves, 
                                          persistence=persistence, 
                                          lacunarity=lacunarity, 
                                          repeatx=width, 
                                          repeaty=height, 
                                          repeatz=depth, 
                                          base=seed)  # Use seed for reproducibility
                    noise_grid[y, x, z] = noise_value
        
        # Normalize and smooth the noise
        noise_grid = (noise_grid - noise_grid.min()) / (noise_grid.max() - noise_grid.min())
        smoothed_noise = gaussian_filter(noise_grid, sigma=sigma)
        
        # Apply threshold to create a binary mask
        binary_mask = (smoothed_noise > threshold).astype(np.uint8)
        return binary_mask

def intersect_noise_with_brain(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract the brain mask and intersect it with a provided binary mask.
    
    Parameters:
    - image: np.ndarray
        A 2D or 3D tensor representing the brain image.
    - mask: np.ndarray
        A binary numpy array representing the mask to intersect with.
        
    Returns:
    - final_mask: np.ndarray
        A tensor representing the intersection of the brain mask and the provided mask.
    """
    # Step 1: Determine the brain mask
    # Binary mask for brain: non-zero regions in the image
    brain_mask = image > 0
    
    # Fill holes in the brain mask to ensure a complete region
    brain_mask = binary_fill_holes(brain_mask)
    
    # Keep the largest connected component (assumes brain is the largest)
    labeled_mask, num_features = label(brain_mask)
    if num_features == 0:
        raise ValueError("No brain region detected in the input image.")
    # Find the largest connected component
    component_sizes = np.bincount(labeled_mask.flat)
    largest_label = np.argmax(component_sizes[1:]) + 1  # Ignore background (label 0)
    brain_mask = labeled_mask == largest_label

    # Step 2: Intersect with the input mask
    intersection_mask = brain_mask & mask
    
    return intersection_mask