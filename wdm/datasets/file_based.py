import lmdb
import numpy as np
import nibabel as nib
import pickle
from abc import abstractmethod
from torch.utils.data import Dataset

from ..configuration.files import LMDBDatasetConfig, NIFTIDatasetConfig, MedicalFileConfig
from ..utils.etc import normalize_image
    
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
    
class NIFTIDataset(MedicalDataset):
     
    def __init__(self, config: NIFTIDatasetConfig):
        super(NIFTIDataset, self).__init__(config=config)

        self.location = config.sourcefile.file_path
        self.paths_with_cov: dict = config.get_files_and_cov()
        self.length = len(self.paths_with_cov)

    def __len__(self):
        return self.length
    
    def get_image(self, path: str):
        img = np.asarray(nib.load(path).dataobj,dtype=float)
        if self.config.is_volume: img = img.transpose((2, 1, 0))
        return normalize_image(img)
    
    def get_sample(self, index: int):
        path, age, sex = self.paths_with_cov[index]
        img = self.get_image(path)
        return img, float(age), sex

    def get_location(self):
        return self.location
