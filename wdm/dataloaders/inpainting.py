from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Generator
from typing import Iterable, Collection

from .mri import MRIDataloader
from ..datasets.file_based import MedicalDataset
from ..datasets.mri import MRIMaskedDataset
from ..utils.masking import MaskSampler
from ..utils.etc import get_default_device

class MRIInpaintDataLoader(MRIDataloader):
    
    def __init__(self,
                 dataset: MedicalDataset | Collection[MedicalDataset],
                 age_range: Iterable[float],
                 mask_sampler: MaskSampler,
                 seed: int = 11111,
                 num_workers: int = 15,
                 batch_size: int = 16,
                 transforms : list[Module] = None,
                 *args,
                 **kwargs):
        # Now explicitly call the parent constructor with all parameters
        super(MRIInpaintDataLoader, self).__init__(dataset=dataset, 
                         age_range=age_range,
                         seed=seed,
                         batch_size=batch_size,
                         *args,
                         **kwargs)
        self.save_hyperparameters(ignore="dataset", logger=False)
        

    def setup(self, stage=None):
        if not self.train_set:
            self.train_set = MRIMaskedDataset(self.dataset, self.hparams.age_range, mask_sampler=self.hparams.mask_sampler, 
                                              transforms=self.hparams.transforms)

    def train_dataloader(self):
        if self.train_set is None:
            self.setup()
        return DataLoader(self.train_set, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
                          shuffle=True, generator=Generator().manual_seed(self.hparams.seed))

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError