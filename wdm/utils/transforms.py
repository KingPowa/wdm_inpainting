import numpy as np
from typing import Callable, List
import torch
import torch.nn.functional as F
from torch.nn import Module
import torchvision.transforms as tf

from .etc import is_iterable

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