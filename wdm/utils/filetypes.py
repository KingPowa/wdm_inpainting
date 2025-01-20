import os
import pandas as pd
import numpy as np
from abc import abstractmethod
from typing import Any, Callable, Optional, Union

class SourceFile:
    def __init__(self, file_path: str):
        self.file_path = file_path

    @property
    def basename(self) -> str:
        return os.path.basename(self.file_path)
    
    @property
    def filename(self) -> str:
        return os.path.splitext(self.basename)[0]
    
    @property
    def modality(self) -> str:
        # Default: inferred
        return self.filename.split("_")[-1]
    
    @abstractmethod
    def get_paths(self) -> list[tuple[str, Any, str]]:
        raise NotImplementedError
    
    def exists(self) -> bool:
        return os.path.exists(self.file_path)
    

class Masterfile(SourceFile):
    def __init__(
        self,
        file_path: str,
        modality: Optional[str] = None,
        path_func: Optional[Union[str, Callable[[pd.DataFrame], pd.Series]]] = None,
        age_func: Optional[Union[str, Callable[[pd.DataFrame], pd.Series]]] = None,
        sex_func: Optional[Union[str, Callable[[pd.DataFrame], pd.Series]]] = None
    ):
        """
        Initializes the Masterfile.

        :param file_path: Path to the masterfile.
        :param modality: Modality identifier.
        :param path_func: Either a string representing the column name for paths,
                          or a function that maps the DataFrame to a Series of paths.
        :param age_func: Either a string representing the column name for age,
                         or a function that maps the DataFrame to a Series of ages.
        :param sex_func: Either a string representing the column name for sex,
                         or a function that maps the DataFrame to a Series of sex values.
        """
        super(Masterfile, self).__init__(file_path=file_path)
        self.modality_prm = modality
        self.path_func = path_func
        self.age_func = age_func
        self.sex_func = sex_func

    def read_pandas_structure(self, path: str) -> pd.DataFrame:
        if path.endswith('.xls') or path.endswith('.xlsx'):
            return pd.read_excel(path)
        else:
            return pd.read_csv(path)
    
    @property
    def modality(self) -> str:
        return self.modality_prm if self.modality_prm else SourceFile.modality.fget(self)
    
    def _apply_func_or_column(self, df: pd.DataFrame, func_or_col: Optional[Union[str, Callable[[pd.DataFrame], pd.Series]]]) -> pd.Series:
        """
        Helper method to apply a function or select a column from the DataFrame.

        :param df: The DataFrame to operate on.
        :param func_or_col: Either a string (column name) or a callable.
        :return: A pandas Series with the desired values.
        """
        if func_or_col is None:
            raise ValueError("A column or function must be provided for path_func, age_func, or sex_func.")
        elif isinstance(func_or_col, str):
            if func_or_col not in df.columns:
                raise ValueError(f"Column '{func_or_col}' not found in DataFrame.")
            return df[func_or_col]
        elif callable(func_or_col):
            result = func_or_col(df)
            if not isinstance(result, pd.Series):
                raise TypeError("Mapping functions must return a pandas Series.")
            return result
        else:
            raise TypeError("path_func, age_func, and sex_func must be either str or callable.")

    def get_subset_paths(self, pths: list) -> list[tuple[Any, Any, Any]]:
        df = self.read_pandas_structure(self.file_path)
        
        # Apply path_func to get the path series
        path_series = self._apply_func_or_column(df, self.path_func)
        
        # Subset the DataFrame based on paths
        subset_df = df[path_series.isin(pths)]
        
        # Apply age_func and sex_func
        age_series = self._apply_func_or_column(df, self.age_func)
        sex_series = self._apply_func_or_column(df, self.sex_func)
        
        # Select the relevant columns
        subset = pd.DataFrame({
            'path': path_series[subset_df.index],
            'age': age_series[subset_df.index],
            'sex': sex_series[subset_df.index]
        })
        
        return self.__make_dict(subset.values)
    
    def get_paths(self) -> list[tuple[Any, Any, Any]]:
        df = self.read_pandas_structure(self.file_path)
        
        # Apply path_func, age_func, and sex_func
        path_series = self._apply_func_or_column(df, self.path_func)
        age_series = self._apply_func_or_column(df, self.age_func)
        sex_series = self._apply_func_or_column(df, self.sex_func)
        
        # Create a subset DataFrame with the required columns
        subset = pd.DataFrame({
            'path': path_series,
            'age': age_series,
            'sex': sex_series
        })
        
        return self.__make_dict(subset.values)
    
    def __make_dict(self, vals: np.ndarray) -> list:
        """
        Converts a NumPy array to a list of tuples.

        :param vals: NumPy array containing the values.
        :return: List of tuples.
        """
        return [tuple(val) for val in vals]


class Nifti(SourceFile):
    def __init__(self, file_path: str, cov_file: Union[str, Masterfile] = None):
        """
        Initializes the Nifti class.

        :param file_path: Path to the Nifti file.
        :param cov_file: Path to the covariate file or an instance of Masterfile.
        """
        super(Nifti, self).__init__(file_path=file_path)
        if cov_file is None and not self.file_path.endswith('sv'):
            raise FileNotFoundError("Covariate file not provided")
        self.cov_file = cov_file

    def exists(self) -> bool:
        cov_exists = os.path.exists(self.cov_file) if isinstance(self.cov_file, str) else (self.cov_file.exists() if self.cov_file else True)
        return super().exists() and cov_exists
    
    def get_paths(self) -> list[tuple[Any, Any, Any]]:
        if self.cov_file:
            # Read paths from the file_path
            with open(self.file_path, 'r') as f:
                pths = [pth.strip() for pth in f.readlines()]
            
            # Handle cov_file being a string or Masterfile instance
            if isinstance(self.cov_file, str):
                masterfile = Masterfile(self.cov_file)
                return masterfile.get_subset_paths(pths)
            else:
                masterfile: Masterfile = self.cov_file
                return masterfile.get_subset_paths(pths)
        else:
            masterfile = Masterfile(self.filename)
            return masterfile.get_paths()