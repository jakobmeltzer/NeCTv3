from __future__ import annotations

import math
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.utils.data
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
import matplotlib.pyplot as plt
from nect.config import Config
from nect.trainers.base_trainer import BaseTrainer

# torch.autograd.set_detect_anomaly(True)  # debug only — disabled for performance


class ContinousScanningTrainer(BaseTrainer):
    def __init__(
        self,
        config: Config,
        output_directory: str | Path | None = None,
        checkpoint: str | Path | None = None,
        **kwargs,
        #save_ckpt: bool = True,
        #save_last: bool = True,
        #save_optimizer: bool = True,
        #verbose: bool = True,
        #log: bool = True,
        #cancel_at: str | None = None,
        #keep_two: bool = True,
        #prune: bool = True,
    ):
        super().__init__(
            config=config,
            output_directory=output_directory,
            checkpoint=checkpoint,
            **kwargs,
            #save_ckpt=save_ckpt,
            #save_last=save_last,
            #save_optimizer=save_optimizer,
            #verbose=verbose,
            #log=log,
            #cancel_at=cancel_at,
            #keep_two=keep_two,
            #prune=prune
        )
        if config.accumulation_steps is None:
            raise ValueError("accumulation_steps must be provided")
        if config.continous_scanning is False:
            raise ValueError("continous_scanning must be True")

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
                        vmin = float(self.dataset.minimum.item())
                        vmax = float(np.percentile(dynamic, 99))
                        axes[1, i].imshow(dynamic, cmap="gray", interpolation="none", vmin=vmin, vmax=vmax)

                    for ax in axes.ravel():
                        ax.set_axis_off()

                    fig.tight_layout()
                else:
                    if size[0] * size[1] * size[2] < self.config.points_per_batch:
                        output = self.model(grid).squeeze().reshape(size).squeeze().detach().cpu().numpy()
                    else:
                        output = torch.zeros(size).numpy()
                        for i in range(size[0]):
                            output[i] = (self.model(grid[i * size[1] * size[2] : (i + 1) * size[1] * size[2]]).squeeze(0).reshape(size[1], size[2]).squeeze(1).detach().cpu().numpy())

                    output = output / self.geometry.max_distance_traveled
                    output = output * (self.dataset.maximum.item() - self.dataset.minimum.item())
                    output = output + self.dataset.minimum.item()
                    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
                    vmin = float(self.dataset.minimum.item())
                    vmax = float(np.percentile(output, 99))
                    axes[0].hist(output.flatten(), bins=100, range=(vmin, vmax))
                    axes[1].imshow(output, cmap="gray", interpolation="none", vmin=vmin, vmax=vmax)
                save_path = f"{self.image_directory_base}/{self.current_epoch:04}_{self.current_angle:04}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()
            self.last_image_time = time.perf_counter()

    def fit(self):
        self.step = 0
        self.training_time = time.perf_counter()
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        for epoch in self.tqdm(range(self.current_epoch, self.config.epochs), total=self.config.epochs, leave=True, desc="Epochs",):
            self._epoch_loss_sum = 0.0
            self._epoch_loss_count = 0
            self.on_train_epoch_start()
            tqdm_bar = self.tqdm(enumerate(self.dataloader), total=len(self.dataloader), leave=False, desc="Projections",)
            for i, (proj, angle_start, angle_stop, timestep) in tqdm_bar:
                if i < self.current_angle:
                    continue

                self.on_angle_start(proj, angle_start)
                memory_info = nvmlDeviceGetMemoryInfo(h)
                if self.verbose:
                    tqdm_bar.set_postfix({"GPU mem%": f"{round(int(memory_info.used)/1024**3, 1)}/{int(memory_info.total)/1024**3}G"})
                    tqdm_bar.refresh()

                for batch_num in range(min(cast(int, self.batch_per_proj), self.projector.batch_per_epoch)):
                    self.optim.zero_grad()
                    self.model.train()
                    end_linspace = np.linspace(float(angle_start.detach().cpu()), float(angle_stop.detach().cpu()), self.config.accumulation_steps + 1, endpoint=True, )
                    linspace = [(end_linspace[k] + end_linspace[k + 1]) / 2 for k in range(self.config.accumulation_steps)]
                    points_per_batch = 5000000  # 5 million points per batch is about the maximum that can be processed at once with tinycudann

                    # Pass 1: accumulate y_pred and y_target without computation graphs to save memory.
                    # With all graphs alive simultaneously, memory scales with accumulation_steps.
                    # Cache projector outputs so Pass 2 doesn't recompute them.
                    # Both y_pred and y_target are averaged across all sub-angles so the loss
                    # compares quantities computed at the same set of ray geometries.
                    y_pred = None
                    y_target = None
                    cached_angles = []  # list of (points_filtered, zero_points_mask, points_shape, distances, ray lengths)
                    with torch.no_grad():
                        for ang in linspace:
                            self.projector.update_angle(ang)
                            points, y = self.projector(batch_num=batch_num, proj=self.proj)
                            if points is None or y is None:
                                cached_angles.append(None)
                                continue
                            zero_points_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
                            points_shape = points.size()
                            if points_shape[1] == 0:
                                self.logger.warning("No points in the batch")
                                cached_angles.append(None)
                                continue
                            points_filtered = points.view(-1, 3)[~zero_points_mask]
                            if points_filtered.size(0) == 0:
                                cached_angles.append(None)
                                continue
                            distances = self.projector.distances
                            ray_lengths = self.projector.ray_lengths
                            cached_angles.append((points_filtered, zero_points_mask, points_shape, distances, ray_lengths))
                            atten_hats = []
                            for points_num in range(0, points_filtered.size(0), points_per_batch):
                                if self.config.mode == "dynamic":
                                    atten_hat = self.model(points_filtered[points_num : points_num + points_per_batch], float(timestep),).squeeze(0)
                                else:
                                    atten_hat = self.model(points_filtered[points_num : points_num + points_per_batch]).squeeze(0)
                                atten_hats.append(atten_hat)
                            atten_hat = torch.cat(atten_hats)
                            processed_tensor = torch.zeros((points_shape[0], points_shape[1], 1), dtype=torch.float32, device=self.fabric.device,).view(-1, 1)
                            processed_tensor[~zero_points_mask] = atten_hat
                            atten_hat = processed_tensor.view(points_shape[0], points_shape[1])
                            contrib = self.apply_forward_model(
                                atten_hat,
                                distances,
                                ray_lengths,
                                self.config.accumulation_steps,
                            )
                            if y_pred is None:
                                y_pred = contrib
                                y_target = y / self.config.accumulation_steps
                            else:
                                y_pred += contrib
                                y_target = y_target + y / self.config.accumulation_steps

                    if y_pred is None or y_target is None:
                        continue

                    # Treat y_pred as a leaf to compute dL/dy_pred without touching model params.
                    y_pred_leaf = y_pred.requires_grad_(True)
                    if self.config.add_poisson:
                        y_pred_for_loss = (y_pred_leaf + torch.poisson(y_pred_leaf * 1e5) / 1e5) / 2
                    else:
                        y_pred_for_loss = y_pred_leaf

                    if self.config.s3im and self.current_projection > self.config.warmup.steps:
                        loss = 0
                        patch_size = min(math.floor(math.sqrt(self.projector.total_detector_pixels)), math.floor(math.sqrt(self.batch_size)),)  # 25x25 patch size, add a parameter later
                        self.fabric.log_dict({"patch_size": patch_size}, step=self.step)
                        loss += self.loss_fn(y_pred_for_loss, y_target)
                        loss += self.s3im_loss(y_pred_for_loss, y_target, patch_size=patch_size)
                    else:
                        loss = self.loss_fn(y_pred_for_loss, y_target)

                    loss.backward()
                    grad_y_pred = y_pred_leaf.grad.detach()

                    # Pass 2: replay each angle using cached projector outputs.
                    # Only one computation graph is alive at a time.
                    for cached in cached_angles:
                        if cached is None:
                            continue
                        points_filtered, zero_points_mask, points_shape, distances, ray_lengths = cached
                        atten_hats = []
                        for points_num in range(0, points_filtered.size(0), points_per_batch):
                            if self.config.mode == "dynamic":
                                atten_hat = self.model(points_filtered[points_num : points_num + points_per_batch], float(timestep),).squeeze(0)
                            else:
                                atten_hat = self.model(points_filtered[points_num : points_num + points_per_batch]).squeeze(0)
                            atten_hats.append(atten_hat)
                        atten_hat = torch.cat(atten_hats)
                        processed_tensor = torch.zeros((points_shape[0], points_shape[1], 1), dtype=torch.float32, device=self.fabric.device,).view(-1, 1)
                        processed_tensor[~zero_points_mask] = atten_hat
                        atten_hat = processed_tensor.view(points_shape[0], points_shape[1])
                        contrib = self.apply_forward_model(
                            atten_hat,
                            distances,
                            ray_lengths,
                            self.config.accumulation_steps,
                        )
                        self.fabric.backward((contrib * grad_y_pred).sum())

                    self.fabric.log_dict(
                        {
                            "loss": loss.item(),
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
                        self.fabric.log_dict(
                            {"skip_alpha_value": self.model.skip_alpha.item()},
                            step=self.step,
                        )
                    if self.config.clip_grad_value is not None:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
                    self.optim.step()
                    self.step += 1
                    if torch.isfinite(loss):
                            self._epoch_loss_sum += float(loss.item())
                            self._epoch_loss_count += 1
                self.on_angle_end()
            self.on_train_epoch_end()
        self.evaluate()
        self.save_model(last=True)
