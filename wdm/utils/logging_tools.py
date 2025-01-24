import os
import logging
import glob
import pickle
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig

from .etc import get_timestamp

class Session:

    def __init__(self, name: str, config: DictConfig):
        self.session_identifier = get_timestamp()
        self.base_dir = config.train_config.base_dir
        self.working_directory = os.path.join(self.base_dir, f"{name}_exec_{self.session_identifier}")
        self.config = config
        self.name = name

        os.makedirs(self.working_directory, exist_ok=True)

    def save(self):
        with open(os.path.join(self.working_directory, "session.pkl"), 'wb') as f:
            pickle.dump(self, f)

    @property
    def log_dir(self):
        log_dir = os.path.join(self.working_directory, self.__get_log_dir())
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def __get_log_dir(self):
        return (self.config.train_config.log_dir if "log_dir" in self.config.train_config.keys() else "logs")
    
    @property
    def ckpt_dir(self):
        ckdir = os.path.join(self.working_directory, self.__get_ckpt_dir())
        os.makedirs(ckdir, exist_ok=True)
        return ckdir
    
    @property 
    def fig_dir(self):
        fig_dir = os.path.join(self.working_directory, "figs")
        os.makedirs(fig_dir, exist_ok=True)
        return fig_dir
    
    def __get_ckpt_dir(self):
        return (self.config.train_config.chkpt_dir if "chkpt_dir" in self.config.train_config.keys() else "checkpoints")
    
    @property
    def devices(self):
        return self.config.train_config.devices if "devices" in self.config.train_config.keys() else 1
    
    @property
    def accelerator(self):
        return self.config.train_config.accelerator if "accelerator" in self.config.train_config.keys() else "auto"
    
    def is_slurm(self):
        return 'SLURM_JOB_ID' in os.environ
    
def restore_session(session_path: str) -> Session:
    with open(session_path, 'rb') as f:
        return pickle.load(f)
    
def get_session_from_checkpoint(checkpoint_path: str) -> str:
    assert os.path.exists(checkpoint_path)
    session_path = os.path.dirname(os.path.dirname(checkpoint_path)) # traverse 2 levels
    return os.path.join(session_path, "session.pkl")

def find_latest_ckpt(checkpoint_folder):
    files = glob.glob(os.path.join(checkpoint_folder, f"*.ckpt"))
    latest_path = None
    latest_mod_time = -1
    for path in files:
        try:
            mod_time = os.path.getmtime(path)  # Get the modification time
            if mod_time > latest_mod_time:
                latest_mod_time = mod_time
                latest_path = path
        except FileNotFoundError:
            continue  # Skip paths that don't exist or cannot be accessed

    return latest_path

def find_latest_checkpoint(execution_folder: str, name: str) -> str:
    files = glob.glob(os.path.join(execution_folder, f"{name}_exec_*"))
    latest_path = None
    latest_timestamp = -1
    for path in files:
        # Split the path by underscore
        parts = path.split('_')
        # Ensure the timestamp is valid
        try:
            timestamp = int(parts[-1])  # Last part is the timestamp
            if timestamp > latest_timestamp:
                expected_path = os.path.join(path, "checkpoints")
                if os.path.exists(expected_path):
                    ckpt_path = find_latest_ckpt(expected_path)
                    if ckpt_path is not None:
                        latest_path = ckpt_path
                        latest_timestamp = timestamp
        except ValueError:
            continue  # Skip paths with invalid timestamps

    return latest_path


class RankZeroLogger(logging.Logger):
    """Logger that only logs messages on rank 0 in distributed training."""

    @rank_zero_only
    def info(self, msg, *args, **kwargs):
        super().info(msg, *args, **kwargs)

    @rank_zero_only
    def warning(self, msg, *args, **kwargs):
        super().warning(msg, *args, **kwargs)

    @rank_zero_only
    def error(self, msg, *args, **kwargs):
        super().error(msg, *args, **kwargs)

    @rank_zero_only
    def debug(self, msg, *args, **kwargs):
        super().debug(msg, *args, **kwargs)

    @rank_zero_only
    def critical(self, msg, *args, **kwargs):
        super().critical(msg, *args, **kwargs)

def setup_logger(session: Session, level=logging.INFO) -> RankZeroLogger:
    """Set up a rank-zero logger with timestamp, ensuring singleton behavior."""

    logger_name = f"{session.name}_Logger"
    
    # Check if logger already exists
    if logger_name in logging.root.manager.loggerDict:
        return logging.getLogger(logger_name)  # Return existing instance

    logger = RankZeroLogger(logger_name)
    logger.setLevel(level)

    # Create a file handler to log to a file
    file_handler = logging.FileHandler(os.path.join(session.log_dir, f"activity_{session.session_identifier}.log"))
    file_handler.setLevel(level)

    # Define the log format with timestamp
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger

class AdvancedWandLogger(WandbLogger):

    def __init__(self, 
                 model,
                 session: Session,
                 project: str = None,
                 version = None, 
                 offline = False,
                 log_model = False,
                 **kwargs):
        
        name = f"{model._get_name()}#{session.session_identifier}"
        save_dir = session.log_dir
        if project is None:
            project = f"{model._get_name()}_project"
        
        super().__init__(name, save_dir, version, offline, None, None, None, project, log_model, None, None, None, **kwargs)

class AdvancedModelCheckpoint(ModelCheckpoint):

    def __init__(self,
                 session: Session,
                 monitor: str,
                 mode: str = "min",
                 filename_suffix: str = "",
                 save_last = None, 
                 save_top_k = 3, 
                 every_n_train_steps = None, 
                 train_time_interval = None, 
                 every_n_epochs = None, 
                 save_on_train_epoch_end = None, 
                 enable_version_counter = True):
        
        dirpath = session.ckpt_dir
        filename = f"{filename_suffix}_" + f"{{epoch:02d}}-{{{monitor}:.4f}}"
        
        super().__init__(dirpath, 
                         filename, 
                         monitor, 
                         False, 
                         save_last, 
                         save_top_k, 
                         False, 
                         mode, 
                         True, 
                         every_n_train_steps, 
                         train_time_interval, 
                         every_n_epochs, 
                         save_on_train_epoch_end, 
                         enable_version_counter)