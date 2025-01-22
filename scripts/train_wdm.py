# Final training fileimport os
import torch
import sys
import os
import hydra

# Add the parent directory to the sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import EarlyStopping

from wdm.utils.masking import DirectorySampler
from wdm.configuration.mask import PresampledMaskConfig
from wdm.utils.transforms import Resize3D, MaskTransform
from wdm.utils.logging_tools import setup_logger, Session, AdvancedModelCheckpoint, AdvancedWandLogger
from wdm.datasets.utils import instantiate_datasets
from wdm.dataloaders.inpainting import MRIInpaintDataLoader

from wdm.model.wdm import WDM
from wdm.model.diffusion.sampler import create_named_schedule_sampler
from wdm.model.utils.etc import create_model_and_diffusion, model_and_diffusion_defaults

from wdm.model.wdm import WDM

def main(config_file: str):
    # Constants
    CONFIG_PATH = config_file
    # Load configuration
    config = OmegaConf.load(CONFIG_PATH)
    # Setup Session
    session = Session("wdm_training", config=config)
    # Setup Logger
    logger = setup_logger(session)
    # Setup training vars
    IMG_SHAPE = config.train_config.img_shape
    # Setup datasets
    datasets = instantiate_datasets(config.datasets, logger)
    mask_sampler = DirectorySampler(PresampledMaskConfig("../masks"))
    dataloader = MRIInpaintDataLoader(dataset=datasets[0:], 
                                      age_range=None,
                                       batch_size=config.train_config.batch_size,
                                       mask_sampler=mask_sampler,
                                       num_workers=config.train_config.num_workers,
                                       transforms={"img": [Resize3D(IMG_SHAPE, "trilinear")], 
                                                   "mask": [Resize3D(IMG_SHAPE, "trilinear"), MaskTransform()]}) 
    # Setup Model
    logger.info(f"Declaring Unet and Diffusion model...")
    args = model_and_diffusion_defaults(config.diffusion_config)
    model, diffusion = create_model_and_diffusion(**args)
    sampler = create_named_schedule_sampler("uniform", diffusion,  maxt=args["diffusion_steps"])
    # Setup Module
    logger.info(f"Declaring WDM model...")
    wdm = WDM(
        model=model,
        diffusion=diffusion,
        batch_size=config.train_config.batch_size,
        in_channels=1,
        microbatch=-1,
        lr=config.train_config.lr,
        log_interval=10,
        img_log_interval=10,
        schedule_sampler=sampler,
        mode="Conditional_always_known_only_healthy",
        label_cond_weight=0
    )
    # Setup Training
    logger.info(f"Starting training. Number of steps: {config.diffusion_config.diffusion_steps}")
    logger.info(f"Setting up Wandbboard")
    wand_logger = AdvancedWandLogger(model, session)
    checkpoint_callback = AdvancedModelCheckpoint(session=session,
                                            filename_suffix='holdout',
                                            monitor='loss/MSE',
                                            mode='min')
    # early_stop_callback = EarlyStopping(
    #     monitor='loss/MSE',
    #     patience=config.train_config.patience,           # Number of epochs with no improvement after which training will be stopped
    #     verbose=True,
    #     mode='min'
    # )
    logger.info(f"Setting up Trainer")
    logger.info(f"TRAINING INFO")
    logger.info(f"-- Devices {torch.cuda.device_count()}")
    logger.info(f"-- Slurm {session.is_slurm()}")
    logger.info(f"-- Accelerator {session.accelerator}")
    trainer = Trainer(
        accelerator=session.accelerator,
        devices=torch.cuda.device_count(),  # Automatically detect available GPUs
        num_nodes=session.config.train_config.num_nodes,  # Number of nodes
        max_steps=config.diffusion_config.diffusion_steps,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=wand_logger,
        callbacks=[checkpoint_callback],# early_stop_callback],
        enable_progress_bar=(not session.is_slurm())
    )
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model)
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig(os.path.join(session.fig_dir, "lr_tuner.png"))

    # # Update the model's learning rate
    # new_lr = lr_finder.suggestion()
    # model.hparams.learning_rate = new_lr
    # logger.info(f"Using suggested LR: {new_lr}")

    logger.info(f"Starting trainer")
    trainer.fit(wdm, datamodule=dataloader)

    performances = trainer.callback_metrics
    logger.info(f"Finished. Total performance: {performances}")

if __name__ == "__main__":
    # Check if the parameter is passed
    if len(sys.argv) != 2:
        print("Usage: python train_wdm.py")
        sys.exit(1)

    # Retrieve the parameter from the command line
    config_file = sys.argv[1]
    main(config_file = config_file)
