# Final training fileimport os
import torch
import sys
import os
import hydra
import argparse

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
from wdm.utils.transforms import Resize3D, MaskTransform, QuantileAndNormalize
from wdm.utils.logging_tools import setup_logger, Session, AdvancedModelCheckpoint, AdvancedWandLogger, restore_session, find_latest_checkpoint, get_session_from_checkpoint
from wdm.datasets.utils import instantiate_datasets
from wdm.dataloaders.inpainting import MRIInpaintDataLoader

from wdm.model.wdm import WDM
from wdm.model.diffusion.sampler import create_named_schedule_sampler
from wdm.model.utils.etc import create_model_and_diffusion, model_and_diffusion_defaults, create_gaussian_diffusion

from wdm.model.wdm import WDM

def train_model(name: str, config_file: str):
    # Constants
    CONFIG_PATH = config_file
    # Load configuration
    config = OmegaConf.load(CONFIG_PATH)
    # Setup Session
    session = Session(name, config=config)
    session.save()
    # Setup Logger
    logger = setup_logger(session)
    # Setup training vars
    IMG_SHAPE = [config.model_config.image_size]*3
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
    args = model_and_diffusion_defaults(config.diffusion_config,
                                        config.model_config,
                                        config.common_config)
    model, diffusion = create_model_and_diffusion(**args)
    args['steps'] = config.sampling_config.sampling_steps
    sampling_diffusion = create_gaussian_diffusion(**args)
    sampler = create_named_schedule_sampler("uniform", diffusion, maxt=1000)
    # Setup Module
    logger.info(f"Declaring WDM model...")
    wdm = WDM(
        model=model,
        session=session,
        diffusion=diffusion,
        sampling_diffusion=sampling_diffusion,
        batch_size=config.train_config.batch_size,
        in_channels=1,
        microbatch=-1,
        lr=config.train_config.lr,
        log_interval=10,
        img_log_interval=50,
        val_interval=2000,
        schedule_sampler=sampler,
        weight_decay=config.train_config.weight_decay,
        mask_weight=config.train_config.mask_weight,
        clip_denoised=config.sampling_config.clip_denoised,
        steps_scheduler=config.sampling_config.steps_scheduler,
        sampling_steps=config.sampling_config.sampling_steps
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
        max_steps=config.train_config.steps,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=wand_logger,
        callbacks=[checkpoint_callback],# early_stop_callback],
        enable_progress_bar=(not session.is_slurm()),
        precision=config.trainer_config.precision
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

def resume_train(checkpoint_path, session_path):
    session = restore_session(session_path)
    logger = setup_logger(session)

    logger.info(f"Restored session {session.session_identifier}, working directory: {session.working_directory}")
    config = session.config

    # Setup Model
    logger.info(f"Declaring Unet and Diffusion model...")
    args = model_and_diffusion_defaults(config.diffusion_config,
                                        config.model_config,
                                        config.common_config)
    model, diffusion = create_model_and_diffusion(**args)
    args['steps'] = config.sampling_config.sampling_steps
    sampling_diffusion = create_gaussian_diffusion(**args)
    sampler = create_named_schedule_sampler("uniform", diffusion, maxt=1000)
    # Setup Module
    logger.info(f"Declaring WDM model...")
    wdm = WDM(
        model=model,
        session=session,
        diffusion=diffusion,
        sampling_diffusion=sampling_diffusion,
        batch_size=config.train_config.batch_size,
        in_channels=1,
        microbatch=-1,
        lr=config.train_config.lr,
        log_interval=10,
        img_log_interval=50,
        val_interval=2000,
        schedule_sampler=sampler,
        weight_decay=config.train_config.weight_decay,
        mask_weight=config.train_config.mask_weight,
        clip_denoised=config.sampling_config.clip_denoised,
        steps_scheduler=config.sampling_config.steps_scheduler,
        sampling_steps=config.sampling_config.sampling_steps
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
        max_steps=config.train_config.steps,
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=wand_logger,
        callbacks=[checkpoint_callback],# early_stop_callback],
        enable_progress_bar=(not session.is_slurm()),
        precision=config.trainer_config.precision
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
    trainer.fit(wdm, ckpt_path=checkpoint_path)

    performances = trainer.callback_metrics
    logger.info(f"Finished. Total performance: {performances}")

def main(name: str, config_file: str, restore_checkpoint: str):
    if restore_checkpoint == "no":
        train_model(name, config_file)
    elif "auto:" in restore_checkpoint:
        checkpoint_path = find_latest_checkpoint(restore_checkpoint.replace("auto:", ""), name)
        session_path = get_session_from_checkpoint()
        resume_train(checkpoint_path, session_path)
    elif os.path.exists(restore_checkpoint):
        resume_train(checkpoint_path, get_session_from_checkpoint())
    else:
        raise ValueError(f"No existent or invalid value of checkpoint path: {restore_checkpoint}")

if __name__ == "__main__":
    # Check if the parameter is passed
    parser = argparse.ArgumentParser(description="WDM training script")

    parser.add_argument('config_path', type=str, help="Path of the configuration file")
    parser.add_argument('--restore_checkpoint', type=str, required=False, help="Path to checkpoint to restore. If specified, it grabs the restore checkpoint. " + 
                        "If 'auto:<path>', it automatically detects the lastest checkpoint that failed in the path", default="no")
    parser.add_argument('--name', type=str, required=False, help="Name of the execution", default="wdm_training")

    args = parser.parse_args()

    config_file = args.config_path
    restore_checkpoint = args.restore_checkpoint
    name = args.name

    # Retrieve the parameter from the command line
    main(name, config_file, restore_checkpoint)
