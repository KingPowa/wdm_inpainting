from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split
from torch import Generator
from torch.nn import Module
from collections.abc import Iterable, Collection

from ..datasets.file_based import MedicalDataset
from ..datasets.mri import MRIDataset
from ..utils.etc import is_iterable

class MRIDataloader(LightningDataModule):

    def __init__(self,
                 dataset: MedicalDataset | Collection[MedicalDataset],
                 age_range: Iterable[float],
                 seed: int = 11111,
                 batch_size: int = 16):
        if age_range:
            assert is_iterable(age_range) and len(age_range) >= 2
            self.age_range = age_range[:2] if age_range[0] < age_range[1] else age_range[:2:-1]
        
        super().__init__()
        self.save_hyperparameters(ignore="dataset", logger=False)

        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.dataset = dataset

    def info(self):
        return {
            "Dataset Name" : self.dataset.get_name(),
            "Modality": self.dataset.modality,
            "Batch Size": self.hparams.batch_size,
            "Seed": self.hparams.seed,
            "Min-Max Age": f"{self.hparams.min_age}-{self.hparams.max_age}"
        }.items()

class MRIHoldoutDataLoader(MRIDataloader):
    def __init__(self,
                 dataset: MedicalDataset | Collection[MedicalDataset],
                 age_range = None,
                 seed: int = 11111,
                 num_workers: int = 15,
                 batch_size: int = 16,
                 train_holdout=0.7,
                 val_holdout=1,
                 transforms:list[Module] = None):
        # Now explicitly call the parent constructor with all parameters
        super().__init__(dataset=dataset, 
                         age_range=age_range, 
                         seed=seed, 
                         batch_size=batch_size)
        self.save_hyperparameters(ignore="dataset", logger=False)
        
    def setup(self, stage=None):
        if not self.train_set and not self.valid_set:
            self.dataset = MRIDataset(self.dataset, self.hparams.age_range, self.hparams.transforms)
            self.train_set, valid_set = random_split(self.dataset, [self.hparams.train_holdout, 1-self.hparams.train_holdout], generator=Generator().manual_seed(self.hparams.seed))
            self.valid_set, self.test_set = random_split(valid_set, [self.hparams.val_holdout, 1-self.hparams.val_holdout])

    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
                          shuffle=True, generator=Generator().manual_seed(self.hparams.seed))

    def val_dataloader(self):
        return DataLoader(self.valid_set, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size)
    
class MRIKFoldDataLoader(MRIDataloader):

    def __init__(self, 
                 dataset: MedicalDataset | Collection[MedicalDataset],
                 age_range: Iterable[float],
                 seed: int = 11111,
                 num_workers: int = 15,
                 batch_size: int = 16,
                 k = 0,
                 folds = 10):
        
        # Now explicitly call the parent constructor with all parameters
        super().__init__(dataset=dataset, 
                         age_range=age_range,
                         seed=seed, 
                         num_workers=num_workers, 
                         batch_size=batch_size)
        assert 1 <= k <= folds, "incorrect fold number"

        self.k = k
        self.folds = folds
        self.save_hyperparameters(ignore="dataset", logger=False)

    def setup(self, stage=None):
        if not self.train_set and not self.valid_set:
            dataset = MRIDataset(dataset, self.hparams.age_range)
            self.train_set, valid_set = random_split(dataset, [self.hparams.train_holdout, 1-self.hparams.train_holdout], generator=Generator().manual_seed(self.hparams.seed))
            self.valid_set, self.test_set = random_split(valid_set, [self.hparams.val_holdout, 1-self.hparams.val_holdout])
    
    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            dataset = MRIDataset(dataset, self.hparams.age_range)

            # choose fold to train on
            kf = KFold(n_splits=self.hparams.folds, shuffle=True, random_state=self.hparams.seed)
            all_splits = [k for k in kf.split(dataset)]
            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            self.data_train, self.data_val = dataset[train_indexes], dataset[val_indexes]

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                          shuffle=True, generator=Generator().manual_seed(self.hparams.seed))

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)