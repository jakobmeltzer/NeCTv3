from __future__ import annotations

import datetime
import logging
import math
import os
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast
import shutil

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from loguru import logger
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
from torchinfo import summary
from tqdm import tqdm
import tinycudann as tcnn

import nect
from nect.data import NeCTDataset
from nect.forward_model import log_projection
from nect.network import KPlanes
from nect.network.kplanes import regularize_k_planes
from nect.utils import create_sub_folders, setup_logger, total_variation_3d, prune_model


if TYPE_CHECKING:
    from nect.config import Config
torch.autograd.set_detect_anomaly(True)


class BaseTrainer:
    def __init__(
        self,
        config: Config,
        output_directory: str | Path | None = None,
        checkpoint: str | Path | None = None,
        save_ckpt: bool = True,
        save_last: bool = True,
        save_optimizer: bool = True,
        verbose: bool = True,
        log: bool = True,
        cancel_at: str | None = None,
        keep_two: bool = True,
        prune: bool = True,
    ):
        setup_logger(level=logging.INFO if verbose else logging.WARNING)
        if checkpoint:
            output_directory = Path(checkpoint).parent.parent.parent
        elif output_directory is not None:
            specific_run = datetime.datetime.now().replace(microsecond=0).isoformat().replace(":","-")
            output_directory = os.path.join(output_directory, config.get_str(), specific_run)
        self.use_prior = config.use_prior
        self.save_optimizer = save_optimizer
        self.keep_two = keep_two
        if self.use_prior:
            prior_path = Path(config.img_path)
            self.prior = np.load(prior_path).astype(np.float32)
        L.seed_everything(42)
        self.logger = logger
        self.logger.info("-------------------")
        self.logger.info(f"SEEING NUMBER OF GPUS {torch.cuda.device_count()}")
        self.logger.info("-------------------")
        torch.set_float32_matmul_precision("high")
        self.cancel_at = None
        self.prune = prune
        if cancel_at is not None:
            try:
                self.cancel_at = datetime.datetime.fromisoformat(cancel_at)
            except ValueError as e:
                raise ValueError(f"Tried set cancel of job at '{cancel_at}', which is an invalid ISO-datetime.")
        self.config = config
        self.setup_dataset()
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
        if isinstance(self.config.epochs, str):
            fraction = self.config.epochs.split("x")
            self.config.epochs = math.ceil(49 / len(self.dataset) * max(config.geometry.nDetector))
            if len(fraction) == 2:
                self.config.epochs = math.ceil(self.config.epochs * float(fraction[0]))
        self.loss_fn = config.get_loss_fn()

        if output_directory is None:
            if save_ckpt or save_last or log:
                raise ValueError("Output directory must be provided if logging or saving checkpoints")
        if log:
            if output_directory is None:
                # raise a error that is due to coding error and not the user
                raise ValueError("Output directory must be provided if logging or saving checkpoints. This is a bug in the code.")
            tensorboard_logger = TensorBoardLogger(root_dir=output_directory, name="logs")
        else:
            self.config.image_interval = -1
            tensorboard_logger = None
        self.verbose = verbose
        if verbose is False:
            self.tqdm = lambda x, *args, **kwargs: x
        else:
            self.tqdm = tqdm
        self.fabric = L.Fabric(
            accelerator="cuda",
            devices="auto",
            strategy="auto",
            precision="16-mixed",
            loggers=tensorboard_logger,
        )

        # supress warnings from self.fabric.launch()
        warnings.filterwarnings("ignore")
        self.fabric.launch()
        warnings.resetwarnings()
        self.model = config.get_model()
        self.optim = config.get_optimizer(self.model)
        (self.lr_scheduler_warmup, self.lr_scheduler, self.lr_scheduler_warmup_downsample,) = config.get_lr_schedulers(self.optim)
        self.current_epoch = 0
        self.current_angle = 0
        self.angle = 0.0
        self.current_projection = 0
        if config.s3im:
            self.s3im_loss = config.get_s3im_loss()

        self.downsample_detector_factor = config.downsampling_detector.start
        self.points_per_ray = config.points_per_ray.start
        self.geometry = nect.Geometry.from_cfg(config.geometry, reconstruction_mode=config.reconstruction_mode, sample_outside=config.sample_outside,)
        if config.points_per_batch == "auto":
            raise ValueError("`points_per_batch` should already have been calculated at this point.")
        self.projector = nect.sampling.Projector(
            geometry=self.geometry,
            points_per_batch=config.points_per_batch // 2,
            points_per_ray=self.points_per_ray,
            device=self.fabric.device,
            uniform_ray_spacing=config.uniform_ray_spacing,
        )

        self.model, self.optim = self.fabric.setup(self.model, self.optim)
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "B"):
            self.model.encoder.B = self.fabric.to_device(self.model.encoder.B)

        self.dataloader = cast(torch.utils.data.DataLoader, self.fabric.setup_dataloaders(self.dataloader))
        if config.points_per_ray.end == "auto":
            config.points_per_ray.end = math.ceil(max(config.geometry.nDetector) * 1.5)
        elif isinstance(config.points_per_ray.end, str):
            end_str = config.points_per_ray.end.split("x")
            if len(end_str) == 2:
                config.points_per_ray.end = math.ceil(max(config.geometry.nDetector) * float(end_str[0]))
            else:
                raise ValueError("Invalid format for `points_per_ray.end`")
        if isinstance(config.points_per_ray.update_interval, str):
            if config.points_per_ray.update_interval == "auto":
                factor = 0.9
            else:
                update_interval = config.points_per_ray.update_interval.split("x")
                if len(update_interval) == 2:
                    factor = float(update_interval[0])
                else:
                    raise ValueError(f"Invalid format for `points_per_ray.update_interval`, got '{config.points_per_ray.update_interval}'")
                
            projections = len(self.dataset)
            epochs = self.config.epochs * factor
            total_updates = projections * epochs / torch.cuda.device_count()
            number_of_updates_needed = config.points_per_ray.end - config.points_per_ray.start
            config.points_per_ray.update_interval = math.ceil(total_updates / number_of_updates_needed)

        if self.fabric.is_global_zero and output_directory is not None:
            self.checkpoint_directory_base, self.image_directory_base = create_sub_folders(output_directory)
            self.epoch_loss_log_path = Path(self.checkpoint_directory_base).parent / "epoch_losses.txt"
            self.initial_state_path = None#Path(self.checkpoint_directory_base).parent / "initial_state.txt"
        else:
            self.checkpoint_directory_base = "needs_to_be_defined_but_not_used"  # must be defined
            self.image_directory_base = "needs_to_be_defined_but_not_used"
            self.epoch_loss_log_path = None
            self.initial_state_path = None

        self._initial_state_saved = False
        self.last_checkpoint_time = time.perf_counter()
        self.last_image_time = time.perf_counter()
        self.last_evaluation_time = time.perf_counter()
        self.save = save_ckpt
        self.save_last = save_last
        self.use_checkpoint = False
        self.downsample_warmup_iteration = 0
        if checkpoint:
            self.logger.info("LOADING CHEKPOINT")
            checkpoint_data = self.fabric.load(checkpoint, strict=True)
            self.current_angle = checkpoint_data["angle"]
            self.current_epoch = checkpoint_data["epoch"]
            self.model.load_state_dict(checkpoint_data["model"])
            self.optim.load_state_dict(checkpoint_data["optim"])
            self.logger.info(f"Starting from epoch {self.current_epoch} and angle number {self.current_angle}")
            self.dataloader._num_iter_calls = self.current_epoch
            self.use_checkpoint = True
            self.current_projection = self.current_epoch * config.geometry.num_angles + self.current_angle
            self.downsample_detector_factor = max(1, self.config.downsampling_detector.start // 2 ** (self.current_projection // self.config.downsampling_detector.update_interval)) 
            if self.config.points_per_ray.linear:
                self.points_per_ray = min(self.config.points_per_ray.end, self.config.points_per_ray.start + self.current_projection // self.config.points_per_ray.update_interval)
            else:
                self.points_per_ray = min(self.config.points_per_ray.end, self.config.points_per_ray.start * 2 ** (self.current_projection // self.config.points_per_ray.update_interval))
            self.projector.update(
                angle=self.angle,
                detector_binning=self.downsample_detector_factor,
                points_per_ray=self.points_per_ray,
                random_offset_detector=0.5 if self.downsample_detector_factor > 1 else 0,
            )
            self.step = 0
            self.lr_scheduler.load_state_dict(checkpoint_data["lr_scheduler"])
        self.setup_evaluator()
        self.batch_per_proj = config.batch_per_proj
        if self.batch_per_proj == "all":
            self.batch_per_proj = 1000000000

        if config.mode == "static" and log:
            model_summary = summary(self.model, input_size=((1, 3)))
            # save the model summary to a txt file
            with open(f"{Path(self.image_directory_base).parent/'model_summary.txt'}", "w") as file:
                file.write(str(model_summary))

        if self.fabric.is_global_zero and checkpoint is None and output_directory is not None:
            config.save(output_directory)
        
        self.outputdir = output_directory
    
    def get_model(self):
        return self.model

    def apply_forward_model(
        self,
        attenuation: torch.Tensor,
        distances: torch.Tensor,
        ray_lengths: torch.Tensor | None = None,
        accumulation_steps: int = 1,
    ) -> torch.Tensor:
        """Map sampled attenuation to a log projection.

        ``distances`` is the integration step length (delta s), while
        ``ray_lengths`` is the total clipped ray length L. Both are normalized
        by ``max_distance_traveled``. When the extension is enabled:

            y = delta_s * sum(mu_i) - beta * L**2.

        ``accumulation_steps`` averages complete sub-angle projections for
        continuous scanning.
        """
        projection = log_projection(
            attenuation,
            distances,
            self.geometry.max_distance_traveled,
            beta=self.model.beta if self.config.use_extended_forward_model else None,
            ray_lengths=ray_lengths,
        )
        return projection / accumulation_steps
    
    def set_model(self, newmodel):
        self.model = newmodel
        
    def get_outputdir(self):
        return self.outputdir

    def _write_vram_stats(self):
        if not self.fabric.is_global_zero:
            return
        if not torch.cuda.is_available():
            return
        model_dir = Path(self.checkpoint_directory_base).parent
        if not model_dir.exists():
            return
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()
        try:
            props = torch.cuda.get_device_properties(0)
            total_bytes = props.total_memory
            gpu_name = props.name
        except Exception:
            total_bytes = 0
            gpu_name = "unknown"
        lines = [
            f"GPU: {gpu_name}",
            f"Peak allocated: {peak_allocated_bytes / 1024**3:.3f} GB",
            f"Peak reserved:  {peak_reserved_bytes / 1024**3:.3f} GB",
            f"Total VRAM:     {total_bytes / 1024**3:.3f} GB",
        ]
        with open(model_dir / "vram.txt", "w") as f:
            f.write("\n".join(lines) + "\n")

    def setup_dataset(self):
        self.dataset = NeCTDataset(config=self.config, device="cpu",)  # if gpu memory is less than 50 GB, load to cpu

    def setup_evaluator(self):
        pass

    def set_projection(self, projection):
        """
        Set the projection. This is used to update the downsampling factor and points per ray.

        Args:
            projection (int): current projection
        """
        if isinstance(self.config.points_per_ray.end, str):
            raise ValueError("`points_per_ray.end` should already have been calculated at this point.")
        self.projection = projection
        if (
            projection != 0
            and projection % self.config.downsampling_detector.update_interval == 0
            and self.downsample_detector_factor > self.config.downsampling_detector.end
        ):
            self.downsample_detector_factor //= 2
        if (
            projection != 0
            and projection % self.config.points_per_ray.update_interval == 0
            and self.points_per_ray < self.config.points_per_ray.end
            and self.config.points_per_ray.linear is False
        ):
            self.points_per_ray *= 2
        elif (
            projection != 0
            and projection % self.config.points_per_ray.update_interval == 0
            and self.points_per_ray < self.config.points_per_ray.end
            and self.config.points_per_ray.linear is True
        ):
            self.points_per_ray += 1

        self.projector.update(
            angle=self.angle,
            detector_binning=self.downsample_detector_factor,
            points_per_ray=self.points_per_ray,
            random_offset_detector=0.5 if self.downsample_detector_factor > 1 else 0,
        )

    def on_train_epoch_start(self):
        if self.use_checkpoint:
            self.use_checkpoint = False
        else:
            self.current_angle = 0

        if(self.current_epoch==0):
            self.generate_image()

    def on_train_epoch_end(self):
        if(self.config.checkpoint_epoch is not None and self.config.checkpoint_epoch > 0 and self.current_epoch % self.config.checkpoint_epoch == 0):
            self.generate_image()

        if self.fabric.is_global_zero and getattr(self, "_epoch_loss_count", 0) > 0 and self.epoch_loss_log_path is not None:
            avg_loss = self._epoch_loss_sum / max(1, self._epoch_loss_count)
            try:
                self.epoch_loss_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.epoch_loss_log_path, "a", encoding="utf-8") as f:
                    elapsed = time.perf_counter() - self.training_time
                    f.write(f"epoch={self.current_epoch}, avg_loss={avg_loss:.6f}, time={elapsed:.1f}s\n")
            except Exception as e:
                self.logger.warning(f"Failed writing epoch loss log: {e}")
                
        self.current_epoch = self.current_epoch + 1

    def on_angle_end(self):
        if self.cancel_at is not None and datetime.datetime.now(datetime.UTC) > self.cancel_at:
            self.save_model(last=True)
            exit()

        self.current_projection = self.current_projection + 1
        self.lr_scheduler.step()
        if self.current_projection <= self.config.warmup.steps:
            self.lr_scheduler_warmup.step()

    def on_angle_start(self, proj, angle):
        self.proj = proj
        self.angle = angle
        self.set_projection(self.current_projection)
        time_since_last_image = time.perf_counter() - self.last_image_time
        if time_since_last_image > self.config.image_interval and self.config.image_interval > 0:
            self.generate_image()

        time_since_last_checkpoint = time.perf_counter() - self.last_checkpoint_time
        if time_since_last_checkpoint > self.config.checkpoint_interval and self.config.checkpoint_interval > 0:
            self.save_model()

        time_last_evaluation = time.perf_counter() - self.last_evaluation_time
        if (
            self.config.evaluation is not None
            and self.config.evaluation.evaluate_interval > 0
            and time_last_evaluation > self.config.evaluation.evaluate_interval
        ):
            self.evaluate()

        if self.downsample_detector_factor != 1:
            if self.proj.dim() == 3:
                self.proj = F.avg_pool2d(
                    self.proj.unsqueeze(0),
                    kernel_size=self.downsample_detector_factor,
                    stride=self.downsample_detector_factor,
                ).squeeze(0)
            elif self.proj.dim() == 4:
                self.proj = F.avg_pool2d(
                    self.proj,
                    kernel_size=self.downsample_detector_factor,
                    stride=self.downsample_detector_factor,
                )
            else:
                raise ValueError(
                    f"Expected projection tensor with 3 or 4 dimensions, "
                    f"got shape {tuple(self.proj.shape)}"
                )
        
        self.proj = self.proj.flatten()
        self.current_angle = self.current_angle + 1

    def evaluate(self):
        pass

    def save_volume(self, save_path: str | None = None) -> None:
        if self.fabric.is_global_zero:
            self.create_volume(save=True, save_path=save_path)

    def generate_image(self, prior: bool = False):
        with torch.no_grad():
            if self.config.points_per_batch == "auto":
                return
            
            plot = self.config.plot_type
            if plot is None:
                return
            
            if self.fabric.is_global_zero:
                size = [*self.config.geometry.nVoxel]
                sample_size = [*size]
                rm = self.config.sample_outside
                if rm > 0:
                    sample_size = [size[0], size[1] + 2 * rm, size[2] + 2 * rm]

                if plot == "XZ":
                    size[1] = 1

                elif plot == "YZ":
                    size[2] = 1

                elif plot == "XY":
                    size[0] = 1

                if size[0] * size[1] * size[2] > self.config.points_per_batch:
                    sample_size = [sample_size[i] // 3 for i in range(3)]
                    sample_size = [s if s > 0 else 1 for s in sample_size]

                z, y, x = torch.meshgrid(
                    [
                        torch.linspace(0, 1, steps=sample_size[0]) if plot != "XY" else torch.tensor(0.5),
                        torch.linspace(0, 1, steps=sample_size[1])[slice(rm, -rm) if rm > 0 else slice(None)] if plot != "XZ" else torch.tensor(0.5),
                        torch.linspace(0, 1, steps=sample_size[2])[slice(rm, -rm) if rm > 0 else slice(None)] if plot != "YZ" else torch.tensor(0.5),
                    ],
                    indexing="ij",
                )
                grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t().to(self.fabric.device)
                size = list(z.shape)
                self.model.eval()
                if self.config.mode == "dynamic":
                    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
                    avg = self.model(grid, torch.tensor(0)).squeeze().reshape(size).squeeze().detach().cpu().numpy()
                    for i in range(3):
                        dynamic = (self.model(grid, torch.tensor((i + 1) / 4)).squeeze().reshape(size).squeeze().detach().cpu().numpy())
                        axes[0, i].imshow(dynamic - avg, cmap="gray", interpolation="none")
                        dynamic = dynamic / (self.geometry.max_distance_traveled * 2)
                        dynamic = dynamic * (self.dataset.maximum.item() - self.dataset.minimum.item())
                        dynamic = dynamic + self.dataset.minimum.item()
                        axes[1, i].imshow(dynamic, cmap="gray", interpolation="none")
                        vmin = float(self.dataset.minimum.item())
                        vmax = float(np.percentile(dynamic, 99))
                        #axes[1, i].imshow(dynamic, cmap="gray", interpolation="none", vmin=vmin, vmax=vmax)

                    for ax in axes.ravel():
                        ax.set_axis_off()

                    fig.tight_layout()
                else:
                    if size[0] * size[1] * size[2] < self.config.points_per_batch:
                        output = self.model(grid).squeeze().reshape(size).squeeze().detach().cpu().numpy()
                    else:
                        output = torch.zeros(size).numpy()
                        for i in range(size[0]):
                            output[i] = (
                                self.model(grid[i * size[1] * size[2] : (i + 1) * size[1] * size[2]])
                                .squeeze(0)
                                .reshape(size[1], size[2])
                                .squeeze(1)
                                .detach()
                                .cpu()
                                .numpy()
                            )

                    output = output / self.geometry.max_distance_traveled
                    output = output * (self.dataset.maximum.item() - self.dataset.minimum.item())
                    output = output + self.dataset.minimum.item()
                    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
                    #axes[0].hist(output.flatten(), bins=100)
                    #axes[1].imshow(output, cmap="gray", interpolation="none")
                    vmin = float(self.dataset.minimum.item())
                    vmax = float(np.percentile(output, 99))
                    axes[0].hist(output.flatten(), bins=100, range=(vmin, vmax))
                    axes[1].imshow(output, cmap="gray", interpolation="none", vmin=vmin, vmax=vmax)
                save_path = f"{self.image_directory_base}/{self.current_epoch:04}_{self.current_angle:04}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()
            self.last_image_time = time.perf_counter()

    def create_volume(self, save=True, save_path: str | None = None, timestep: float | None = None, cpu=False):
        if self.config.mode == "dynamic" and timestep is None:
            return
        
        with torch.no_grad():
            if self.fabric.is_global_zero:
                size = tuple([*self.config.geometry.nVoxel])
                rm = self.config.sample_outside
                sample_size = [size[0], size[1] + 2 * rm, size[2] + 2 * rm]
                output = torch.zeros(size, dtype=torch.float32, device=self.fabric.device if not cpu else "cpu")
                z_lin = torch.linspace(
                    0.5 / size[0],
                    1 - 0.5 / size[0],
                    steps=size[0],
                    device=self.fabric.device,
                )
                for i in range(size[0]):
                    z, y, x = torch.meshgrid(
                        [z_lin[i],
                        torch.linspace(0.5 / sample_size[1], 1 - 0.5 / sample_size[1], steps=sample_size[1], device=self.fabric.device,)[slice(rm, -rm) if rm > 0 else slice(None)],
                        torch.linspace(0.5 / sample_size[2], 1 - 0.5 / sample_size[2], steps=sample_size[2], device=self.fabric.device,)[slice(rm, -rm) if rm > 0 else slice(None)],
                        ],
                        indexing="ij",
                    )
                    grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).t()
                    if self.config.mode == "static":
                        output_slice = self.model(grid).reshape(size[1], size[2])
                    else:
                        output_slice = self.model(grid, timestep).reshape(size[1], size[2])

                    output_slice = output_slice / self.geometry.max_distance_traveled
                    output_slice = output_slice * (self.dataset.maximum.item() - self.dataset.minimum.item())
                    output_slice = output_slice + self.dataset.minimum.item()
                    output[i] = output_slice if not cpu else output_slice.cpu()
                if save:
                    if save_path is None:
                        save_path = f"{self.image_directory_base}/final.npy"
                    np.save(f"{self.image_directory_base}/final.npy", output.cpu().numpy())

                return output.float()

    def save_model(self, last=False):
        if self.save or (last and self.save_last):
            if self.save_optimizer:
                state = {
                    "model": self.model,
                    "optim": self.optim,
                    "epoch": self.current_epoch,
                    "angle": self.current_angle,
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                }
            else:
                state = {
                    "model": self.model,
                    "epoch": self.current_epoch,
                    "angle": self.current_angle,
                }
            if self.keep_two:
                last_ckpt = os.path.join(self.checkpoint_directory_base, "last.ckpt")
                if os.path.exists(last_ckpt):
                    second_last_ckpt = os.path.join(self.checkpoint_directory_base, "2nd_last.ckpt")
                    if os.path.exists(second_last_ckpt):
                        os.remove(second_last_ckpt)
                    shutil.move(last_ckpt, second_last_ckpt)
            self.logger.info("Saving model - time might take some time")
            self.fabric.save(os.path.join(self.checkpoint_directory_base, "last.ckpt"), state)
            self.logger.info("Saving model finished")
            self.create_volume(self.config.save_volume)
            self.last_checkpoint_time = time.perf_counter()
        elif last:
            self.create_volume(self.config.save_volume)
            self.last_checkpoint_time = time.perf_counter()
        if last and self.prune and self.fabric.is_global_zero:
            prune_model(self.model, Path(self.checkpoint_directory_base).parent)
    
    def save_epoch_checkpoint(self):
        """Save full model + optimizer every N epochs into outputs/run_name/checkpoints/epoch_xxxx.ckpt"""
        run_name = getattr(self.config, "model", "default_run")
        base_dir = Path("outputs") / run_name / "checkpoints"
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = base_dir / f"epoch_{self.current_epoch:04d}.ckpt"

        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict() if self.save_optimizer else None,
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if hasattr(self, "lr_scheduler") else None,
            "config": self.config,
        }

        torch.save(state, filename)
        self.logger.info(f"Saved checkpoint: {filename}")
    
    def _save_initial_parameters_text(self):
        """
        Save the entire model parameter vector as plain text (one number per line),
        with NO extra text. Runs only on global rank zero.
        """
        if self._initial_state_saved:
            return
        if not (self.fabric.is_global_zero and self.initial_state_path is not None):
            return

        try:
            # Flatten all parameters into a single 1D tensor (preserve dtypes)
            with torch.no_grad():
                vec = torch.nn.utils.parameters_to_vector([p.detach() for p in self.model.parameters()])

            # Move to CPU numpy for text write; keep numeric-only content
            np_arr = vec.detach().cpu().numpy()

            # Ensure parent dir exists
            self.initial_state_path.parent.mkdir(parents=True, exist_ok=True)

            # Write ONLY numbers (no headers/footers), one per line
            # Using numpy tofile with sep + format avoids any extra text.
            np_arr.tofile(self.initial_state_path, sep="\n", format="%.18e")

            self._initial_state_saved = True
        except Exception as e:
            # Don't crash training on logging failure; warn instead.
            self.logger.warning(f"Failed to save initial_state.txt: {e}")

    def fit(self):
        try:
            self.step = 0
            self.training_time = time.perf_counter()
            self._save_initial_parameters_text()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)

            for epoch in self.tqdm(range(self.current_epoch, self.config.epochs), total=self.config.epochs, initial=self.current_epoch, leave=True, desc="Epochs",):
                self.on_train_epoch_start()
                self._epoch_loss_sum = 0.0
                self._epoch_loss_count = 0
                tqdm_bar = self.tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False, desc="Projections",)
                for i, (proj, angle, timestep) in tqdm_bar:
                    if i < self.current_angle:
                        continue

                    self.on_angle_start(proj, angle)
                    memory_info = nvmlDeviceGetMemoryInfo(h)
                    if self.verbose:
                        tqdm_bar.set_postfix({"GPU mem%": f"{round(int(memory_info.used)/1024**3, 1)}/{int(memory_info.total)/1024**3}G"})
                        tqdm_bar.refresh()

                    for batch_num in range(min(cast(int, self.batch_per_proj), self.projector.batch_per_epoch)):
                        self.optim.zero_grad()
                        self.model.train()
                        points, y = self.projector(batch_num=batch_num, proj=self.proj)
                        if points is None or y is None:
                            continue

                        zero_points_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
                        points_shape = points.size()
                        if points_shape[1] == 0:
                            self.logger.warning("No points in the batch")
                            continue

                        points = points.view(-1, 3)[~zero_points_mask]
                        if points.size(0) == 0:
                            continue

                        atten_hats = []
                        points_per_batch = 5000000  # 5 million points per batch is about the maximum that can be processed at once with tinycudann
                        for points_num in range(0, points.size(0), points_per_batch):
                            if self.config.mode == "dynamic":
                                atten_hat = self.model(points[points_num : points_num + points_per_batch], float(timestep),).squeeze(0)  # .view((points.size(0), points.size(1)))
                            else:
                                atten_hat = self.model(points[points_num : points_num + points_per_batch]).squeeze(0)  # .view((points.size(0), points.size(1)))
                            atten_hats.append(atten_hat)

                        atten_hat = torch.cat(atten_hats)
                        processed_tensor = torch.zeros((points_shape[0], points_shape[1], 1), dtype=torch.float32, device=self.fabric.device,).view(-1, 1)
                        processed_tensor[~zero_points_mask] = atten_hat
                        atten_hat = processed_tensor.view(points_shape[0], points_shape[1])
                        y_pred = self.apply_forward_model(
                            atten_hat,
                            self.projector.distances,
                            self.projector.ray_lengths,
                        )
                        if self.config.add_poisson:
                            y_pred = (y_pred + torch.poisson(y_pred * 1e5) / 1e5) / 2

                        if self.config.s3im and self.current_projection > self.config.warmup.steps:
                            loss = 0
                            patch_size = min(math.floor(math.sqrt(self.projector.total_detector_pixels)), math.floor(math.sqrt(self.batch_size)),)  # 25x25 patch size, add a parameter later
                            self.fabric.log_dict({"patch_size": patch_size}, step=self.step)
                            loss += self.loss_fn(y_pred, y, i)
                            loss += self.s3im_loss(y_pred, y, patch_size=patch_size)
                        else:
                            loss = self.loss_fn(y_pred, y, i)
                            """
                        if self.use_prior:
                            z, y, x = torch.meshgrid(
                                [
                                    torch.linspace(
                                        0, 1, steps=1000,
                                        device=self.fabric.device,
                                    ),
                                    torch.linspace(
                                        0, 1, steps=1000,
                                        device=self.fabric.device,
                                    ),
                                    torch.linspace(
                                        0, 1, steps=1000,
                                        device=self.fabric.device,
                                    ),
                                ],
                                indexing="ij",
                            )
                            grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).view(3, -1).t()
                            points_dry = self.model(grid, 0).view(1000, 1000 ,1000)
                            points_wet = self.model(grid, 1).view(1000, 1000 ,1000)
                            loss += torch.functional.loss.l1(self.prior_volume, points_dry)
                            loss += torch.functional.loss.l1(self.prior_volume, points_wet)
                            """
                        if isinstance(self.model, KPlanes):
                            regularize_k_planes(self.config.encoder, self.model)
                        self.fabric.log_dict(
                            {
                                "loss": loss,
                                "max_mem": torch.cuda.max_memory_allocated(),
                                "current_mem": int(memory_info.used),
                                "epoch": epoch,
                                "downsample_detector_factor": self.downsample_detector_factor,
                                "points_per_ray": self.points_per_ray,
                                "distance_between_points": self.projector.distance_between_points,
                                **(
                                    {"beta": self.model.beta.detach()}
                                    if self.config.use_extended_forward_model
                                    else {}
                                ),
                                "lr": self.optim.param_groups[0]["lr"],
                                "num_proj_processed": self.current_projection,
                            },
                            step=self.step,
                        )
                        if hasattr(self.model, "skip_alpha"):
                            self.fabric.log_dict({"skip_alpha_value": self.model.skip_alpha.item()}, step=self.step,)
                            
                        if (self.config.mode == "static" and self.current_projection > self.config.warmup.steps and self.config.tv > 0):
                            rand_zyx = np.random.rand(3) * 0.8
                            z, y, x = torch.meshgrid([
                                    torch.linspace(rand_zyx[0], rand_zyx[0] + 0.2, steps=100, device=self.fabric.device,),
                                    torch.linspace(rand_zyx[1], rand_zyx[1] + 0.2, steps=100, device=self.fabric.device,),
                                    torch.linspace(rand_zyx[2], rand_zyx[2] + 0.2, steps=100, device=self.fabric.device,),
                                    ],
                                indexing="ij",
                            )
                            grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).view(3, -1).t()
                            grid = grid.to(self.fabric.device)
                            atten_hat = self.model(grid).squeeze(0)  # .view((points.size(0), points.size(1)))
                            atten_hat = atten_hat.view(100, 100, 100)
                            tv_loss = total_variation_3d(atten_hat, weight=self.config.tv)
                            loss += tv_loss

                        if (self.config.mode == "dynamic" and self.current_projection > self.config.warmup.steps and self.config.tv_temporal > 0):
                            t_float = float(timestep)
                            t_step = 1.0 / max(1, self.dataset.num_timesteps - 1)
                            t_perturbed = min(1.0, t_float + t_step)
                            tv_grid = torch.rand(500, 3, device=self.fabric.device)
                            f_t = self.model(tv_grid, t_float)
                            f_tp = self.model(tv_grid, t_perturbed)
                            temporal_tv_loss = self.config.tv_temporal * torch.mean(torch.abs(f_tp - f_t))
                            loss += temporal_tv_loss

                        if (self.config.mode == "dynamic" and self.current_projection > self.config.warmup.steps and self.config.tv_spatial > 0):
                            t_float = float(timestep)
                            rand_zyx = np.random.rand(3) * 0.8
                            z, y, x = torch.meshgrid([
                                    torch.linspace(rand_zyx[0], rand_zyx[0] + 0.2, steps=50, device=self.fabric.device),
                                    torch.linspace(rand_zyx[1], rand_zyx[1] + 0.2, steps=50, device=self.fabric.device),
                                    torch.linspace(rand_zyx[2], rand_zyx[2] + 0.2, steps=50, device=self.fabric.device),
                                    ],
                                indexing="ij",
                            )
                            grid = torch.stack((z.flatten(), y.flatten(), x.flatten())).view(3, -1).t()
                            atten_hat = self.model(grid, t_float).squeeze(-1)
                            atten_hat = atten_hat.view(50, 50, 50)
                            tv_loss = total_variation_3d(atten_hat, weight=self.config.tv_spatial)
                            loss += tv_loss

                        self.fabric.backward(loss)
                        if self.config.clip_grad_value is not None:
                            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)

                        if torch.isfinite(loss):
                            self._epoch_loss_sum += float(loss.item())
                            self._epoch_loss_count += 1

                        self.optim.step()
                        self.step += 1

                    self.on_angle_end()

                self.on_train_epoch_end()

            self.evaluate()
            self.save_model(last=True)
            self._write_vram_stats()

        except KeyboardInterrupt:
            if self.step > 3000:
                self.logger.info("Please wait before canceling again as the model now is beeing saved")
                self.save_model(last=True)
            else:
                self.logger.info(f"The model was trained for {self.step} steps, so it will not save the model before exiting.")

    def warmup_w0_only(self, steps: int = 1500, lr_mult: float = 2.0, include_b0: bool = True):
        """
        Short warm-up training that updates ONLY the first MLP layer's weights (W0)
        and optionally its bias (b0). Everything else (encoders and deeper MLP layers)
        is frozen via a gradient mask on the flat tcnn 'net.params' tensor.

        Args:
            steps: number of optimizer updates to run in this warm-up (not epochs)
            lr_mult: learning-rate multiplier relative to self.config.base_lr
            include_b0: if True, also train the first layer bias b0
        """
        self.logger.info(f"[W0 warm-up] steps={steps}, lr_mult={lr_mult}, include_b0={include_b0}")

        # ---- Helpers to compute MLP splits (TCNN-compatible; folds padding into W0) ----
        def _mlp_layer_splits(in_dim: int, net_cfg) -> list[int]:
            # H, L + concrete dict
            if hasattr(net_cfg, "n_neurons"):
                H = int(net_cfg.n_neurons)
                L = int(net_cfg.n_hidden_layers)
                net_conf = net_cfg.get_network_config()
            else:
                net_conf = net_cfg.get_network_config()
                H = int(net_conf["n_neurons"])
                L = int(net_conf["n_hidden_layers"])
            D_in = int(in_dim)
            D_out = 1  # single scalar output

            splits: list[int] = []
            splits += [H * D_in, H]                 # W0, b0
            for _ in range(L - 1):
                splits += [H * H, H]                # Wk, bk
            splits += [D_out * H, D_out]            # W_out, b_out

            # Validate vs dummy and fold any padding into W0
            enc = {"otype": "Identity", "n_dims_to_encode": D_in}
            dummy = tcnn.NetworkWithInputEncoding(
                n_input_dims=D_in,
                n_output_dims=D_out,
                encoding_config=enc,
                network_config=net_conf,
            )
            flat = dummy.state_dict().get("net.params", dummy.state_dict().get("params"))
            assert flat is not None, "TCNN dummy state_dict missing 'net.params'/'params'."
            diff = flat.numel() - sum(splits)
            if diff != 0:
                splits[0] += diff
            assert sum(splits) == flat.numel(), (
                f"MLP split mismatch even after padding W0: sum={sum(splits)} vs tcnn={flat.numel()}"
            )
            return splits

        def _encoded_width_quadcubes_from_cfg(cfg) -> int:
            # No include_identity handling as requested
            L = cfg.encoder.n_levels
            F = cfg.encoder.n_features_per_level
            return 4 * (L * F)

        # ---- Locate the flat TCNN parameter and build a W0/b0 mask ----
        # Find the flat parameter (usually named 'net.params')
        name_to_param = dict(self.model.named_parameters())
        flat_name = None
        for k in name_to_param:
            if k.endswith("net.params") or k == "params" or k.endswith(".params"):
                flat_name = k
                break
        if flat_name is None:
            raise RuntimeError("Could not find TCNN flat parameter ('net.params') in model.named_parameters().")

        flat_param: torch.nn.Parameter = name_to_param[flat_name]
        total_len = flat_param.numel()

        # Compute MLP splits using encoded input width from cfg
        in_dim = _encoded_width_quadcubes_from_cfg(self.config)
        splits = _mlp_layer_splits(in_dim, self.config.net)

        mlp_total = sum(splits)
        enc_total = total_len - mlp_total
        if enc_total < 0:
            raise RuntimeError(f"Computed negative encoder size: total={total_len}, mlp={mlp_total}")

        # Offsets inside MLP tail
        def _prefix_offsets(szs: list[int]) -> list[int]:
            offs = [0]
            for s in szs:
                offs.append(offs[-1] + s)
            return offs

        off = _prefix_offsets(splits)
        W0_lo, W0_hi = enc_total + off[0], enc_total + off[1]   # first weight block
        b0_lo, b0_hi = enc_total + off[1], enc_total + off[2]   # first bias block

        # Build mask: True for trainable indices, False elsewhere
        mask = torch.zeros(total_len, dtype=torch.bool, device=flat_param.device)
        mask[W0_lo:W0_hi] = True
        if include_b0:
            mask[b0_lo:b0_hi] = True

        self.logger.info(
            f"[W0 warm-up] total={total_len}, enc_total={enc_total}, "
            f"W0=[{W0_lo},{W0_hi}), b0=[{b0_lo},{b0_hi}) trainable={int(mask.sum().item())}"
        )

        # Register grad mask hook on the flat param
        def _grad_mask_hook(grad: torch.Tensor) -> torch.Tensor:
            g = grad
            if g.is_sparse:
                g = g.to_dense()
            g = g.masked_fill(~mask, 0)
            return g

        hook_handle = flat_param.register_hook(_grad_mask_hook)

        # Temporary optimizer for the flat param only
        base_lr = getattr(self.config, "base_lr", 1e-3)
        lr = float(base_lr) * float(lr_mult)
        # Try to read beta settings from config; fall back to (0.9, 0.95)
        beta1 = getattr(getattr(self.config, "optimizer", object()), "beta1", 0.9)
        beta2 = getattr(getattr(self.config, "optimizer", object()), "beta2", 0.95)
        wd = getattr(getattr(self.config, "optimizer", object()), "weight_decay", 0.0)
        warmup_optim = torch.optim.Adam([flat_param], lr=lr, betas=(beta1, beta2), weight_decay=wd)

        # ---- Run a minimal inner loop for `steps` optimizer updates ----
        self.model.train()
        steps_done = 0
        dataloader_iter = iter(self.dataloader)

        start_t = time.perf_counter()
        while steps_done < steps:
            try:
                proj, angle, timestep = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                proj, angle, timestep = next(dataloader_iter)

            # Optional downsampling, then flatten
            if self.downsample_detector_factor != 1:
                proj = F.avg_pool2d(
                    proj.unsqueeze(0),
                    kernel_size=self.downsample_detector_factor,
                    stride=self.downsample_detector_factor,
                ).squeeze(0)
            proj = proj.flatten()

            # Ensure projector has per-angle sizing (sets batch_size, distances, batch_per_epoch, etc.)
            self.angle = float(angle) if torch.is_tensor(angle) else float(angle)
            self.projector.update(
                angle=self.angle,
                detector_binning=self.downsample_detector_factor,
                points_per_ray=self.points_per_ray,
                random_offset_detector=0.5 if self.downsample_detector_factor > 1 else 0,
            )

            # Exactly one projector batch per step (config.batch_per_proj == 1)
            batch_num = 0
            warmup_optim.zero_grad(set_to_none=True)

            points, y = self.projector(batch_num=batch_num, proj=proj)
            if points is None or y is None:
                continue

            zero_points_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
            points_shape = points.size()
            if points_shape[1] == 0:
                continue

            pts_flat = points.view(-1, 3)[~zero_points_mask]
            if pts_flat.size(0) == 0:
                continue

            # Forward chunking for TCNN comfort (~5M points per chunk)
            atten_hats = []
            ppb = 5_000_000
            for p0 in range(0, pts_flat.size(0), ppb):
                if self.config.mode == "dynamic":
                    atten_hat = self.model(pts_flat[p0:p0+ppb], float(timestep)).squeeze(0)
                else:
                    atten_hat = self.model(pts_flat[p0:p0+ppb]).squeeze(0)
                atten_hats.append(atten_hat)

            atten_hat = torch.cat(atten_hats) if atten_hats else torch.empty(0, device=self.fabric.device)

            processed = torch.zeros(
                (points_shape[0], points_shape[1], 1),
                dtype=torch.float32,
                device=self.fabric.device,
            ).view(-1, 1)
            processed[~zero_points_mask] = atten_hat
            atten_hat = processed.view(points_shape[0], points_shape[1])

            y_pred = self.apply_forward_model(
                atten_hat,
                self.projector.distances,
                self.projector.ray_lengths,
            )
            loss = self.loss_fn(y_pred, y, 0)

            self.fabric.backward(loss)
            if getattr(self.config, "clip_grad_value", None) is not None:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)

            warmup_optim.step()
            steps_done += 1

            if self.fabric.is_global_zero and steps_done % 50 == 0:
                self.fabric.log_dict(
                    {
                        "w0_warmup/loss": loss.detach(),
                        "w0_warmup/steps_done": steps_done,
                        "w0_warmup/lr": lr,
                    },
                    step=steps_done,
                )

        dt = time.perf_counter() - start_t
        self.logger.info(f"[W0 warm-up] finished {steps_done} steps in {dt:.1f}s")

        # Clean up hook & tmp optimizer
        hook_handle.remove()
        for pg in warmup_optim.param_groups:
            pg["lr"] = 0.0
        del warmup_optim
        torch.cuda.empty_cache()


