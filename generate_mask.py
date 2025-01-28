from wdm.configuration.mask import PerlinNoiseConfig
from wdm.utils.masking import generate_perlin_mask_with_contour_smoothing
import numpy as np
from tqdm import tqdm
import os
import sys
from multiprocessing import Pool, cpu_count

def create_single_mask(args):
    """
    Function to generate and save a single Perlin noise mask.

    Parameters:
        args: Tuple containing (i, config, img_size, dir, name, K)

    Returns:
        str: The filename of the saved mask (for tracking progress).
    """
    i, config, img_size, dir, name, K = args
    np.random.seed(i)  # Ensure different seeds per process

    while True:
        # Sample a new threshold for each mask
        threshold = np.random.uniform(0.5, 0.6)
        mask_config = config
        mask_config.threshold = threshold  # Use a new config object to avoid race conditions

        # Generate the mask
        mask: np.ndarray = generate_perlin_mask_with_contour_smoothing(img_size, **mask_config.to_dict(), seed=i)
        h, w, d = img_size
        central_region = mask[K:h-K, K:w-K, K:d-K]

        # Ensure there's enough activation in the central region
        if np.sum(central_region) > K * 3 * 0.05:
            os.makedirs(dir, exist_ok=True)  # Ensure directory exists
            filename = os.path.join(dir, name.format(i=i))
            np.save(filename, mask)
            return filename  # Return the saved file for progress tracking

def create_masks_parallel(mask_config: PerlinNoiseConfig, img_size, dir, name="mask_{i}", n_samples=10, K=200):
    """
    Generate multiple Perlin noise masks in parallel using multiprocessing.

    Parameters:
        mask_config (PerlinNoiseConfig): Configuration object.
        img_size (tuple): Image dimensions.
        dir (str): Directory to save masks.
        name (str): Naming format for files.
        n_samples (int): Number of masks to generate.
        K (int): Padding for central region constraint.
    """
    # Prepare arguments for parallel execution
    args_list = [(i, mask_config, img_size, dir, name, K) for i in range(n_samples)]

    # Use multiprocessing Pool to speed up processing
    num_workers = min(cpu_count(), 8)  # Use all available cores, up to 8 for efficiency
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(create_single_mask, args_list), total=n_samples))

    print(f"Generated {len(results)} masks.")

def main(location: str, samples: int):
    config = PerlinNoiseConfig(
        scale=45,
        octaves=6,
        persistence=0.3,
        lacunarity=3.5,
        threshold=0.6,  # Initial value, overridden per mask
        sigma=1,
        per_sample=1
    )

    # Run parallel mask generation
    create_masks_parallel(config, (240, 240, 240), location, n_samples=samples)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_mask.py <location> <samples=4000>")
        sys.exit(1)

    location = sys.argv[1]
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    main(location, samples)