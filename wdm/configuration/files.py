import os
from abc import abstractmethod
from torchvision import transforms
from collections.abc import Collection, Callable
from typing import Optional

from ..utils.transforms import Nop, Compose
from ..utils.filetypes import SourceFile

class MedicalFileConfig:

    def __init__(self, is_volume: bool = True):
        self.modality = None
        self.mtransforms = Nop()
        self.is_volume = is_volume

    @abstractmethod
    def get_name(self):
        raise NotImplementedError

class NIFTIDatasetConfig(MedicalFileConfig):

    def __init__(self,
                 sourcefile: SourceFile,
                 modality : str = "T1",
                 return_slices: bool = False,
                 name: str = "MRIDataset", 
                 is_volume: bool = True,
                 mtransforms: Optional[Collection[Callable]] | Optional[Callable] = []):
        """Provide interface for masterfiles and nifti files

        Args:
            nifti_file (str): A file specifying the location of each nifti file of a precise modality, with covariates.
            modality (str, optional): modality of the LMDB. It is used to concat to the folder. Defaults inferred from last name before ext.
            name (str, optional): name of the dataset
            is_volume (bool, optional): if the dataset outputs volumes
            mtransforms (Collection, optional): set of transformations to apply. Defaults to [].
            return_slices (bool, optional): if set, it returns images as slices. Defaults to False.
        Raises:
            FileNotFoundError: Not found the lmdb file
            ValueError: Incorrect parameters provided
        """
        super(NIFTIDatasetConfig, self).__init__(is_volume=is_volume)

        if sourcefile is not None and not sourcefile.exists():
            raise FileNotFoundError(f"File {sourcefile} ({type(sourcefile).__name__}) does not exist!")
        elif sourcefile is not None:
            self.sourcefile = sourcefile

        self.paths = None
        self.non_ext = None

        self.get_paths()

        modality = self.sourcefile.modality if not modality else modality
        
        if mtransforms is None or len(mtransforms) < 0:
            mtransforms = Nop()
        else:
            mtransforms = Compose(mtransforms)
        
        self.mtransforms = mtransforms
        self.name = name
        self.modality = modality
        self.return_slices = return_slices

    def get_conf(self):
        return {
            "lmdb_file": self.lmdb_file,
            "mtransforms": self.mtransforms
        }
    
    def get_name(self):
        return self.name
    
    def get_paths(self):
        if self.paths is None:
            self.paths = self.sourcefile.get_paths()
            self.non_ext = self.sanity_check()
            self.paths = [path for path in self.paths if path[0] not in self.non_ext]
        return self.paths
    
    def sanity_check(self):
        non_ext = []
        if self.sourcefile:
            paths = self.sourcefile.get_paths()
            non_ext = [path[0] for path in paths if not os.path.exists(path[0])]

        if len(non_ext) > 0:
            with open("missing_niis.txt", 'w') as f:
                f.writelines([l+'\n' for l in non_ext])
            print(f"Found {len(non_ext)} non-existent nifti files. Dumped to missing_niis.txt")
        
        return non_ext

    def get_files_and_cov(self):
        return self.get_paths()     

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