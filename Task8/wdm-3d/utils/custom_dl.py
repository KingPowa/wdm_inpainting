import lmdb
import numpy as np
import nibabel as nib
import pickle
from abc import abstractmethod
from torch.utils.data import Dataset
from torchvision import transforms
from collections.abc import Collection, Callable
from typing import Optional
import numpy as np
from typing import Callable, List
import torch
import torch.nn.functional as F
from torch.nn import Module
import torchvision.transforms as tf


def is_iterable(the_element, cls=None):
    try:
        iter(the_element)
    except TypeError:
        return False
    else:
        return True if cls is None else not isinstance(the_element, cls)

class Nop(Callable):
    def __call__(self, x: np.ndarray):
        return x
    
class Compose(Callable):

    def __init__(self, tr: List[Callable] = []):
        if tr is None:
            self.tr = []
        else: self.tr = tr

    def __call__(self, x: np.ndarray):
        for t in self.tr:
            x = t(x)
        return x

import numpy as np
from scipy.ndimage import zoom

class Cropper:
    def __init__(self, target_size=(224, 224, 224)):
        self.target_size = target_size

    # def __call__(self, img: np.ndarray) -> np.ndarray:
    #     if len(img.shape) < 3:
    #         raise ValueError("Input volume must have at least 2 dimensions to crop.")
        
    #     z,y,x = img.shape[-3:]
    #     cropz, cropy, cropx = self.target_size
    #     startx = x//2-(cropx//2)
    #     starty = y//2-(cropy//2)
    #     startz = z//2-(cropz//2)
    #     return img[...,startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) < 2:
            raise ValueError("Input volume must have at least 2 dimensions to crop.")
        
        # Ensure the number of dimensions in target_size matches the last dimensions of the input image
        if len(self.target_size) > len(img.shape):
            raise ValueError("Target size dimensions exceed input image dimensions.")
        
        # Get the shape of the image and the target size
        target_size = self.target_size
        img_shape = img.shape

        # Initialize a list to store slice objects
        slices = [slice(None)] * len(img_shape)

        # For the last N dimensions, calculate the crop starting points and the target sizes
        for i in range(1, len(target_size) + 1):
            size = target_size[-i]  # target size for the current dimension
            dim_len = img_shape[-i]  # length of the current dimension

            if size < dim_len:
            
                # Calculate start and end indices for cropping
                start = (dim_len - size) // 2
                end = start + size
                
                # Update the slice for the current dimension
                slices[-i] = slice(start, end)

            else:

                slices[-i] = slice(0, dim_len)

        # Apply the slices to crop the image
        return img[tuple(slices)]
    
class QuantileAndNormalize:

    def __call__(self, img: np.ndarray) -> np.ndarray:
        normalize=(lambda x: 2*x - 1)
        out_clipped = np.clip(img, np.quantile(img, 0.001), np.quantile(img, 0.999))
        out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
        out_normalized= normalize(out_normalized)
        #img = convert_to_tensor(out_normalized, track_meta=False)
        return out_normalized

class WDMAdapter:

    def __init__(self, target_size=(224, 224)):
        """
        Initialize the VolumeAdapter with a target width and height.
        
        Parameters:
        - target_size: Tuple (width, height) to resize the volume to.
        """
        self.target_size = target_size

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Adapt the input volume to the desired dimensions and padding.
        
        Parameters:
        - volume: A 3D numpy array of shape (depth, height, width).
        
        Returns:
        - Adapted volume as a 3D numpy array.
        """
        # Ensure the input is a 3D tensor
        if len(volume.shape) < 3:
            raise ValueError("Input volume must have 3 or more dimensions: (D, H, W).")

        height, width = volume.shape[-2:]

        # Resize width and height to target dimensions
        target_height, target_width = self.target_size
        
        # Compute the scaling factors
        scale_height = target_height / height
        scale_width = target_width / width
        
        # Use the average of height and width scaling factors for depth scaling
        scale_depth = (scale_height + scale_width) / 2

        # Resize the volume
        resized_volume = zoom(volume, 
                              zoom=[1 for _ in volume.shape[:-3]] + [scale_depth, scale_height, scale_width], 
                              order=3)  # Using cubic interpolation
        
        # Get the new depth
        new_depth = resized_volume.shape[0]

        pad_depth = (4 - (new_depth % 4)) % 4
        if pad_depth > 0:
            padding = [(0,0) for _ in volume.shape[:-3]] + [(0, pad_depth), (0, 0), (0, 0)]
            adapted_volume = np.pad(resized_volume, padding, mode='constant', constant_values=0)
        else:
            adapted_volume = resized_volume
        
        return adapted_volume

class VolumeAdapter:

    def __init__(self, target_size=(224, 224)):
        """
        Initialize the VolumeAdapter with a target width and height.
        
        Parameters:
        - target_size: Tuple (width, height) to resize the volume to.
        """
        self.target_size = target_size

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Adapt the input volume to the desired dimensions and padding.
        
        Parameters:
        - volume: A 3D numpy array of shape (depth, height, width).
        
        Returns:
        - Adapted volume as a 3D numpy array.
        """
        # Ensure the input is a 3D tensor
        if len(volume.shape) < 3:
            raise ValueError("Input volume must have 3 or more dimensions: (D, H, W).")

        height, width = volume.shape[-2:]

        # Resize width and height to target dimensions
        target_height, target_width = self.target_size
        
        # Compute the scaling factors
        scale_height = target_height / height
        scale_width = target_width / width
        
        # Use the average of height and width scaling factors for depth scaling
        scale_depth = (scale_height + scale_width) / 2

        # Resize the volume
        resized_volume = zoom(volume, 
                              zoom=[1 for _ in volume.shape[:-3]] + [scale_depth, scale_height, scale_width], 
                              order=3)  # Using cubic interpolation
        
        # Get the new depth
        new_depth = resized_volume.shape[0]

        pad_depth = (4 - (new_depth % 4)) % 4
        if pad_depth > 0:
            padding = [(0,0) for _ in volume.shape[:-3]] + [(0, pad_depth), (0, 0), (0, 0)]
            adapted_volume = np.pad(resized_volume, padding, mode='constant', constant_values=0)
        else:
            adapted_volume = resized_volume
        
        return adapted_volume
    

class Resize3D(Module):
    def __init__(self, size, interpolation="trilinear"):
        """
        Initializes the transform with the target size.
        
        Args:
            size (tuple): A tuple of three integers (D, H, W) specifying the target dimensions.
            interpolation (str): A value that tells which kind of interpolation to use
        """
        super().__init__()
        if not is_iterable(size) or len(size) != 3:
            raise ValueError("Size must be a tuple of three integers (D, H, W).")
        self.size = tuple(size)
        self.interpolation = interpolation

    def forward(self, volume):
        """
        Resizes a 3D volume to the target size.
        
        Args:
            volume (torch.Tensor): A 3D tensor of shape (C, D, H, W) or (D, H, W).
        
        Returns:
            torch.Tensor: The resized volume.
        """
        if not isinstance(volume, torch.Tensor):
            raise TypeError("Input volume must be a torch.Tensor.")
        
        original_dim = volume.dim()
        while 5 - volume.dim() > 0:
            volume = volume.unsqueeze(0)

        # Use trilinear interpolation for resizing
        resized: torch.Tensor = F.interpolate(volume, size=self.size, mode=self.interpolation, align_corners=False)
        while resized.dim() > original_dim:
            resized = resized.squeeze(0)
        return resized

    @staticmethod
    def suggest_optimal_size(size, factor=1.0, ensure=16):
        """
        Suggests a size where all dimensions are divisible by ensure, considering a scaling factor.
        
        Args:
            size (tuple): A tuple of three integers (D, H, W) specifying the original size.
            factor (float): A factor to scale the size before suggesting an optimal size.
            ensure (int); the multiplying factor to ensure.
        
        Returns:
            tuple: Suggested size (D', H', W') with all dimensions divisible by ensure.
        """
        if factor <= 0:
            raise ValueError("Factor must be greater than zero.")
        
        def nearest_multiple_of_ensure(dim, ensure):
            lower = (dim // ensure) * ensure
            upper = lower + ensure
            return lower if abs(lower - dim) < abs(upper - dim) else upper

        # Scale the size using the factor, ensure integer values, then round to nearest multiple of ensure
        scaled_size = tuple(int(round(dim / factor)) for dim in size)
        optimal_size = tuple(nearest_multiple_of_ensure(dim, ensure) for dim in scaled_size)

        return optimal_size
    
class MaskTransform(Module):

    def forward(self, tensor):
        return torch.where(tensor != 0, torch.tensor(1), tensor)

class MedicalFileConfig:

    def __init__(self, is_volume: bool = True):
        self.modality = None
        self.mtransforms = Nop()
        self.is_volume = is_volume

    @abstractmethod
    def get_name(self):
        raise NotImplementedError

class LMDBDatasetConfig(MedicalFileConfig):

    def __init__(self, lmdb_folder : str,
                 modality : str = "T1", 
                 name: str = "MRIDataset", 
                 is_volume: bool = True,
                 mtransforms: Optional[Collection[Callable]] | Optional[Callable] = []):
        """_summary_

        Args:
            lmdb_folder (str): location of the LMDB folder.
            lmdb_file (str, optional): Plain location of the lmbd file. Defaults to None.
            modality (str, optional): modality of the LMDB. It is used to concat to the folder. Defaults to "T1".
            type_img (str, optional): slices or volume. Defaults to "slices".
            name (str, optional): name of the dataset
            is_volume (bool, optional): if the dataset outputs volumes
            mtransforms (Collection, optional): set of transformations to apply. Defaults to [].

        Raises:
            FileNotFoundError: Not found the lmdb file
            ValueError: Incorrect parameters provided
        """
        super(LMDBDatasetConfig, self).__init__(is_volume=is_volume)

        self.type_img = "volumes" if is_volume else "slices"

        if lmdb_folder is not None and modality is not None and self.type_img is not None:
            self.lmdb_folder = os.path.join(lmdb_folder, f"{self.type_img}_{modality}")
            if not os.path.exists(self.lmdb_folder):
                raise FileNotFoundError(f"File {self.lmdb_folder} does not exist!")
        else:
            raise ValueError(f"Incorrect parameter for LMDB provided.")
        
        if mtransforms is None or len(mtransforms) < 0:
            mtransforms = Nop()
        else:
            mtransforms = Compose(mtransforms)
        
        self.mtransforms = mtransforms
        self.name = name
        self.modality = modality

    def get_conf(self):
        return {
            "lmdb_file": self.lmdb_folder,
            "mtransforms": self.mtransforms
        }
    
    def get_name(self):
        return self.name
    
class MedicalDataset(Dataset):

    def __init__(self, config: MedicalFileConfig):
        super().__init__()
        self.config = config
        self.transforms = config.mtransforms

    @abstractmethod
    def get_sample(self, index: int):
        pass

    def __getitem__(self, index: int) -> tuple[np.ndarray, float, str]:
        tensor, age, sex = self.get_sample(index)
        tensor = self.transforms(tensor) # Apply any useful numpy based transform
        return tensor, age, sex # Unsqueezing to add modality info

    def get_name(self):
        return self.config.get_name()
    
    @abstractmethod
    def get_location(self):
        pass
    
    @property
    def modality(self):
        return self.config.modality
    
class CombinedMedicalDataset(MedicalDataset):

    def __init__(self, datasets: List[MedicalDataset]):
        self.datasets = datasets
        self.concat_dataset = ConcatDataset(datasets)

    def __getitem__(self, index: int) -> tuple[np.ndarray, float, str]:
        dataset_idx, sample_idx = self._get_dataset_and_index(index)
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return len(self.concat_dataset)

    def _get_dataset_and_index(self, index: int):
        """ Helper function to map global index to corresponding dataset and local index. """
        current_len = 0
        for i, dataset in enumerate(self.datasets):
            if index < current_len + len(dataset):
                return i, index - current_len
            current_len += len(dataset)
        raise IndexError("Index out of range")

    def get_sample(self, index: int):
        """ Retrieves a sample using the dataset-specific get_sample() method. """
        dataset_idx, sample_idx = self._get_dataset_and_index(index)
        return self.datasets[dataset_idx].get_sample(sample_idx)

    def get_name(self):
        """ Combines the names of all datasets. """
        return "_".join([ds.get_name() for ds in self.datasets])

    def get_location(self):
        """ Returns locations of all datasets. """
        return [ds.get_location() for ds in self.datasets]

    @property
    def modality(self):
        """ Assumes all datasets share the same modality or returns a list if they differ. """
        modalities = list(set(ds.modality for ds in self.datasets))
        return modalities[0] if len(modalities) == 1 else modalities

# Custom Dataset for loading data from a generic LMDB file
class LMDBDataset(MedicalDataset):
    def __init__(self, config: LMDBDatasetConfig):
        super(LMDBDataset, self).__init__(config=config)
        self.lmdb_folder = config.lmdb_folder
        self.type_img = config.type_img[:-1]

        self.env = lmdb.open(self.lmdb_folder, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __open_lmdb(self):
        """Open an lmdb file specified in the constructor
        """
        self.env = lmdb.open(
            self.lmdb_folder,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)
    
    def get_sample(self, index: int):
        if not hasattr(self, "txn"):
            self.__open_lmdb()
        byteflow = self.txn.get(f"{self.type_img}_{index:08}".encode("ascii"))
        try:
            unpacked = pickle.loads(byteflow)
        except Exception as e:
            print(index)
            raise e

        return unpacked[self.type_img], float(unpacked["age"]), unpacked["sex"]
    
    def get_location(self):
        return self.lmdb_folder

from torchvision import transforms as tr
from torch.utils.data import Dataset, ConcatDataset

from typing import Optional, Collection

import torch
import numpy as np
from noise import pnoise2, pnoise3
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_fill_holes, label

import os
import numpy as np
import random
from abc import abstractmethod

class MaskConfig:

    def __init__(self, per_sample):
        self.per_sample = per_sample

    def __iter__(self):
        # Return the instance attributes as a dictionary for unpacking
        return iter(self.__dict__.items())

    def to_dict(self):
        # Return a dictionary of initialized attributes
        return {key: value for key, value in self.__dict__.items() if key in self.__class__.__init__.__code__.co_varnames and key != "per_sample"}

class PerlinNoiseConfig(MaskConfig):

    def __init__(self, 
                 scale=100, 
                 octaves=6, 
                 persistence=0.5, 
                 lacunarity=2.0,
                 threshold=0.5, 
                 sigma=1.0,
                 per_sample=10):
        super(PerlinNoiseConfig, self).__init__(per_sample=per_sample)
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.threshold = threshold
        self.sigma = sigma

class PresampledMaskConfig(MaskConfig):

    def __init__(self, directory, per_sample=10):
        super(PresampledMaskConfig, self).__init__(per_sample=per_sample)
        self.directory = directory

class MultiplePatchConfig(MaskConfig):

    def __init__(self):
        pass

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
    
class MRIDataset(Dataset):

    def __init__(self, 
                 dataset: MedicalDataset | Collection[MedicalDataset], 
                 age_range: Collection[float] = None,
                 transforms: Optional[Collection[torch.nn.Module]] | Optional[torch.nn.Module] = []):
        if age_range:
            assert is_iterable(age_range) and len(age_range) >= 2
            self.age_range = age_range[:2] if age_range[0] < age_range[1] else age_range[:2:-1]
        else: self.age_range = None

        self.age_range = None
        
        self.dataset = dataset if not is_iterable(dataset, cls=MedicalDataset) else CombinedMedicalDataset(dataset)
        self.length = len(self.dataset)
        self.mtransforms = tr.Compose(transforms) if transforms else torch.nn.Identity()

    def __len__(self):
        return self.length
    
    def get_sample(self, index: int):
        slice, age, sex = self.dataset[index]
        # Slice is Modalities x Channels x Width X Length. We add 1 x W x L to Channels
        # Normalize the integer value
        if self.age_range:
            normalized_age = (age - self.age_range[0]) / (self.age_range[1] - self.age_range[0])
        else: 
            normalized_age = age
        # Convert gender to binary (0 for M, 1 for F)
        sex_binary = 0 if sex == 'M' else 1
        return slice, torch.tensor([normalized_age, sex_binary])

    def __getitem__(self, index: int):
        image, cond = self.get_sample(index)
        image = torch.from_numpy(image).float()
        image = self.mtransforms(image)

        return image.unsqueeze(0), cond # Add channel information

class MRIMaskedDataset(MRIDataset):

    def __init__(self, 
                 dataset: MedicalDataset, 
                 age_range: Collection[float] = None,
                 mask_sampler: MaskSampler = None,
                 transforms: Optional[dict[str, torch.nn.Module]] = {}):
        super().__init__(dataset=dataset, age_range=age_range, transforms=None)

        self.mtransforms = tr.Compose(transforms["img"]) if (transforms and 'img' in transforms) else torch.nn.Identity()
        self.mmtransforms = tr.Compose(transforms["mask"]) if (transforms and 'mask' in transforms) else torch.nn.Identity()
        self.mask_sampler = mask_sampler
        self.length = self.length * mask_sampler.config.per_sample

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slice, cond = self.get_sample(index // self.mask_sampler.config.per_sample)
        slice_shape = slice.shape # (H, W)
        # mask = np.zeros((1))
        # Consider deterministic salt in the configuration to account for 0 masks, until it is non 0
        # salt = 0
        # Note: seed = index since this is the EXTENDED index, counting both the num of samples that the same sample
        # while mask.all(arr=0):
        mask = self.mask_sampler.sample(index)
        # Resize mask
        mask = Cropper(slice_shape[-3:])(mask)
        # Adapt to image
        mask = intersect_noise_with_brain(slice, mask)

        # Add channel
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(slice).float()
        image = self.mtransforms(image)
        mask = torch.from_numpy(mask).float()
        mask = self.mmtransforms(mask)

        btch = {
            "t1n": image.unsqueeze(0),
            "mask_healthy": 1-mask,  # If enabled
            "mask_unhealthy": mask,  # If enabled
            "index": index // self.mask_sampler.config.per_sample
        }

        return btch
    
class MRIMaskedDatasetVal(Dataset):

    def __init__(self, 
                 dataset: MRIMaskedDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        btch = self.dataset[index]

        new_btch = {
            "mask": btch["mask_unhealthy"],
            "t1n_voided": btch["t1n"] * (1-btch["mask_unhealthy"]),
            "index": btch["index"]
        }

        return new_btch
    
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Generator
from typing import Iterable, Collection

class MRIInpaintDataLoader(DataLoader):
    
    def __init__(self,
                 dataset: MedicalDataset | Collection[MedicalDataset],
                 age_range: Iterable[float],
                 mask_sampler: MaskSampler,
                 seed: int = 11111,
                 num_workers: int = 15,
                 batch_size: int = 16,
                 transforms : list[Module] = None):
        # Now explicitly call the parent constructor with all parameters

        ds = MRIMaskedDataset(dataset, age_range, mask_sampler=mask_sampler, 
                                              transforms=transforms)
        super(MRIInpaintDataLoader, self).__init__(
            dataset=ds,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True
        )

class MRIInpaintDataLoaderVal(DataLoader):
    
    def __init__(self,
                 dataset: MedicalDataset | Collection[MedicalDataset],
                 age_range: Iterable[float],
                 mask_sampler: MaskSampler,
                 seed: int = 11111,
                 num_workers: int = 15,
                 batch_size: int = 16,
                 transforms : list[Module] = None):
        # Now explicitly call the parent constructor with all parameters

        

        ds_m = MRIMaskedDataset(dataset, age_range, mask_sampler=mask_sampler, 
                                              transforms=transforms)
        ds = MRIMaskedDatasetVal(ds_m)
        super(MRIInpaintDataLoaderVal, self).__init__(
            dataset=ds,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True
        )

        self.original_dataset = ds_m

from omegaconf import ListConfig, DictConfig
from logging import Logger
import hydra

def instantiate_datasets(datasets: ListConfig[DictConfig], logger: Logger = None):
    inst_datasets = []
    if logger:
        logger.info("Selected datasets:")
    else:
        print("Selected datasets:")
    for dataset in datasets:
        inst_dataset: MedicalDataset = hydra.utils.instantiate(dataset)
        inst_datasets.append(inst_dataset)
        if logger:
            logger.info(f"-- {inst_dataset.get_name()} (Location: {inst_dataset.get_location()})")
        else:
            print(f"-- {inst_dataset.get_name()} (Location: {inst_dataset.get_location()})")
    return inst_datasets 

def prepare_dataloader(config):

    datasets = instantiate_datasets(config.datasets)
    mask_sampler = DirectorySampler(PresampledMaskConfig("/mnt/d/Programmazione/Progetti/phd/masks"))

    IMG_SHAPE = [config.model_config.image_size]*3
    dataloader = MRIInpaintDataLoader(dataset=datasets[0:], 
                                    age_range=None,
                                    batch_size=config.train_config.batch_size,
                                    mask_sampler=mask_sampler,
                                    num_workers=config.train_config.num_workers,
                                    transforms={"img": [Resize3D(IMG_SHAPE, "trilinear")], 
                                                "mask": [Resize3D(IMG_SHAPE, "trilinear"), MaskTransform()]}) 
    
    return dataloader

def prepare_dataloader_val(config):

    datasets = instantiate_datasets(config.datasets)
    mask_sampler = DirectorySampler(PresampledMaskConfig("/mnt/d/Programmazione/Progetti/phd/masks"))

    IMG_SHAPE = [config.model_config.image_size]*3
    dataloader = MRIInpaintDataLoaderVal(dataset=datasets[0:], 
                                    age_range=None,
                                    batch_size=config.train_config.batch_size,
                                    mask_sampler=mask_sampler,
                                    num_workers=config.train_config.num_workers,
                                    transforms={"img": [Resize3D(IMG_SHAPE, "trilinear")], 
                                                "mask": [Resize3D(IMG_SHAPE, "trilinear"), MaskTransform()]}) 
    
    return dataloader

def get_original_images(ds: MRIMaskedDataset, indices) -> np.ndarray:
    image_batch = []
    for index in indices:
        img, _, _ = ds.dataset.get_sample(index)
        image_batch.append(img)

    return np.concatenate(image_batch, axis=0)