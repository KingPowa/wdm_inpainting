# This py implements the WDM model by scratch
import time
from typing import Any
import wandb
import torch as th
import torch.utils.tensorboard
from torch.optim import AdamW
from pytorch_lightning import LightningModule

from ..utils.logging_tools import Session
from .logger import *
from .diffusion.diffproc import SpacedDiffusion
from .diffusion.sampler import LossAwareSampler, UniformSampler, ScheduleSampler
from .diffusion.noise import Noise, NormalNoise

NAMES = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
INITIAL_LOG_LOSS_SCALE = 20.0

class WDM(LightningModule):

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        session: Session,
        diffusion: SpacedDiffusion,
        batch_size: int,
        in_channels: int,
        microbatch: int,
        img_log_interval: int,
        lr: float,
        log_interval: int,
        schedule_sampler : ScheduleSampler = None,
        weight_decay: float = 0.0,
        mask_weight: float = 10,
        noise_generator: Noise = NormalNoise()
    ):
        
        super(WDM, self).__init__()

        self.save_hyperparameters(ignore=["model"])

        configure(dir=session.log_dir)
        
        # This is the training modality
        self.model = model
        self.diffusion = diffusion

        self.batch_size = batch_size
        self.in_channels = in_channels

        # As I understood, this control a internal loop logic
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        
        # https://arxiv.org/pdf/1412.6980 7.2, not used
        # Used to reduce overfitting on last fold
        # self.ema_rate = (
        #     [ema_rate]
        #     if isinstance(ema_rate, float)
        #     else [float(x) for x in ema_rate.split(",")]
        # )

        self.log_interval = log_interval
        self.img_log_interval = img_log_interval
        
        # Automatic PL checkpoint
        # self.resume_checkpoint = resume_checkpoint
        
        # Managed by PL
        # self.use_fp16 = use_fp16
        # if self.use_fp16:
        #     self.grad_scaler = amp.GradScaler()
        # else:
        #     self.grad_scaler = amp.GradScaler(enabled=False)

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.noise_generator = noise_generator
        self.mask_weight = mask_weight

        # How much you train. Honestly they set it to 0 so no clue.
        # Anyway we automatise this as the "epochs" on PL trainer
        # self.lr_anneal_steps = lr_anneal_steps

        self.sync_cuda = th.cuda.is_available()

        if not th.cuda.is_available():
            warn("Training requires CUDA.")

        self.automatic_optimization = False
        self.t = time.time()

    @property
    def global_batch(self):
        return self.batch_size * self.trainer.world_size
    
    @property
    def timestep(self):
        tmp = self.t if self.t is not None else time.time()
        self.t = time.time()
        return tmp

    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return opt
    
    def training_step(self, batch, batch_idx):
        t_load = time.time() - self.timestep # Set time

        lossmse, sample, sample_idwt = self._step(batch, self.noise_generator) # Step single batch

        t_fwd = time.time() - self.timestep

        self.log('time/load', t_load, on_step=True)
        self.log('time/forward', t_fwd, on_step=True)
        self.log('time/total', t_load + t_load, on_step=True)
        self.log('loss/MSE', lossmse.detach().item(), on_step=True)

        if self.global_step % self.img_log_interval == 0:
            image_size = sample_idwt.size()[2]
            midplane = sample_idwt[0, 0, :, :, image_size // 2]
            self.logger.experiment.log({'sample/x_0': wandb.Image(midplane.unsqueeze(0))},
                                            step=self.global_step)

            image_size = sample.size()[2]
            for ch in range(8):
                midplane = sample[0, ch, :, :, image_size // 2]
                self.logger.experiment.log({'sample/{}'.format(NAMES[ch]): wandb.Image(midplane.unsqueeze(0))},
                                                step=self.global_step)

        if self.global_step % self.log_interval == 0:
            dumpkvs()

        return lossmse

    def _step(self, batch, noise_generator=None):
        info = dict()
        
        lossmse, sample, sample_idwt = self.forward_backward(batch, noise_generator)
        
        # compute norms
        with torch.no_grad():
            param_max_norm = max([p.abs().max().item() for p in self.model.parameters()])
            grad_max_norm = max([p.grad.abs().max().item() for p in self.model.parameters()])
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not torch.isfinite(lossmse): #infinite
            if not torch.isfinite(torch.tensor(param_max_norm)):
                error(f"model parameters contain non-finite value {param_max_norm}, entering breakpoint", level=ERROR)
                breakpoint()
            else:
                warn(f"model parameters are finite, but loss is not: {lossmse.detach()}"
                           "\n -> update will be skipped in grad_scaler.step()", level=WARN)

        opt = self.optimizers()
        
        opt.step()
        #self._anneal_lr()
        self.log_step()
        return lossmse, sample, sample_idwt            

    def forward_backward(self, batch, noise_generator=None):
        for p in self.model.parameters():  # Zero out gradient
            p.grad = None

        # That's the "miniloop"
        # Basically they divide the batches in more little batches...
        for i in range(0, batch[0].shape[0], self.microbatch):
            micro = (batch_elem[i: i + self.microbatch] for batch_elem in batch)
            img, mask, cond = micro

            t, weights = self.schedule_sampler.sample(img.shape[0], self.device) # Sample schedule sampler

            diffusion_loss = self.diffusion.training_losses(self.model,
                                               x_start=img,
                                               t=t,
                                               model_kwargs=None,
                                               mask=mask,
                                               noise_generator=noise_generator)

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, diffusion_loss["loss"].detach()
                )

            losses: dict[str, torch.Tensor] = diffusion_loss[0]         # Loss value # terms
            sample = diffusion_loss[1]         # Denoised subbands at t=0
            sample_idwt = diffusion_loss[2]    # Inverse wavelet transformed denoised subbands at t=0
            
            if "mse_wav" in losses:
                # Log wavelet level loss
                self.log_wav_loss_prop("mse_wav", losses)

                if float(self.mask_weight)!=0:
                    # Add the mse_label_cond
                    self.log_wav_loss_prop("mse_label_cond", losses)
                    
                    loss = (losses["mse_wav"]).mean() + (int(self.mask_weight)*(losses["mse_label_cond"].mean()))
                else:
                    # weights = th.ones(len(losses["mse_wav"])).cuda()  # Equally weight all wavelet channel losses 
                    loss = (losses["mse_wav"]).mean()

            elif "mse_loss" in losses:
                self.log('loss/mse_loss', losses["mse_loss"][0].item(), on_step=True)
                if float(self.mask_weight)!=0:
                    loss = (losses["mse_loss"]).mean() + (int(self.mask_weight)*(losses["mse_label_cond"].mean()))
                    # Add the mse_label_cond
                    self.log('loss/mse_label_cond', losses["mse_label_cond"][0].item(),
                                                on_step=True)
                else:
                    loss = (losses["mse_loss"]).mean() 
            else:
                loss = (int(self.mask_weight)*(losses["mse_label_cond"].mean()))
                # Add the mse_label_cond
                self.log('loss/mse_label_cond', losses["mse_label_cond"][0].item(),
                                            on_step=True)

            # perform some finiteness checks
            if not torch.isfinite(loss):
                self.txt_logger.log(f"Encountered non-finite loss {loss}")
            else:
                loss.backward()

            self.log_loss_dict(t, {k: v * weights for k, v in losses.items()})

            return loss, sample, sample_idwt
    
    def log_wav_loss_prop(self, prop, losses: dict[str, torch.Tensor]):
        for i,val in enumerate([name.lower() for name in NAMES]):
            self.log(f"loss/{prop}_{val}", losses[f"{prop}_{val}"][i].item(), on_step=True)

    def log_step(self):
        logkv("step", self.global_step)
        logkv("samples", (self.global_step + 1) * self.global_batch)

    def log_loss_dict(self, ts, losses):
        for key, values in losses.items():
            logkv_mean(f"{key}_mean", values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / self.diffusion.num_timesteps)
                logkv_mean(f"{key}_q{quartile}_mean", sub_loss)


