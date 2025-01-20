from omegaconf import ListConfig, DictConfig
from logging import Logger
from .file_based import MedicalDataset
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