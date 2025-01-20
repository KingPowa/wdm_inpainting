import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as tr

from typing import Optional, Collection

from .file_based import MedicalDataset
from ..utils.etc import is_iterable
from ..utils.masking import MaskSampler, intersect_noise_with_brain
from ..utils.transforms import Cropper
    
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
        
        self.dataset = dataset if not is_iterable(dataset, cls=MedicalDataset) else ConcatDataset(dataset)
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
        return image.unsqueeze(0), mask, cond