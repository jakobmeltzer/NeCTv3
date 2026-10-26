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
from nect.trainers.continous_scanning_trainer import ContinousScanningTrainer


class ContinousScanningTrainerMem(ContinousScanningTrainer):
    """
    Continuous scanning trainer that keeps all accumulation-step computation graphs
    alive simultaneously and calls backward() once. Trades GPU memory for faster
    training (accumulation_steps forward passes + 1 backward vs. the base class's
    2 × accumulation_steps forwards + accumulation_steps backwards).
    """

    def fit(self):
        self.step = 0
        self.training_time = time.perf_counter()
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        mem_info = nvmlDeviceGetMemoryInfo(h)
        total_gb = mem_info.total / 1024**3
        print(f"[GPU] Total memory: {total_gb:.1f} GB")
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
                    end_linspace = np.linspace(float(angle_start.detach().cpu()), float(angle_stop.detach().cpu()), self.config.accumulation_steps + 1, endpoint=True,)
                    linspace = [(end_linspace[k] + end_linspace[k + 1]) / 2 for k in range(self.config.accumulation_steps)]
                    points_per_batch = 5000000

                    # Single pass: accumulate y_pred with live computation graphs.
                    # All graphs remain alive until backward(), so peak memory scales
                    # with accumulation_steps. In return, each angle is only forwarded once.
                    y_pred = None
                    y_target = None

                    for ang in linspace:
                        self.projector.update_angle(ang)
                        points, y = self.projector(batch_num=batch_num, proj=self.proj)
                        if points is None or y is None:
                            continue
                        zero_points_mask = torch.all(points.view(-1, 3) == 0, dim=-1)
                        points_shape = points.size()
                        if points_shape[1] == 0:
                            self.logger.warning("No points in the batch")
                            continue
                        points_filtered = points.view(-1, 3)[~zero_points_mask]
                        if points_filtered.size(0) == 0:
                            continue

                        distances = self.projector.distances
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
                            atten_hat, distances, self.config.accumulation_steps
                        )

                        if y_pred is None:
                            y_pred = contrib
                            y_target = y / self.config.accumulation_steps
                        else:
                            y_pred = y_pred + contrib
                            y_target = y_target + y / self.config.accumulation_steps

                    if y_pred is None or y_target is None:
                        continue

                    if self.config.add_poisson:
                        y_pred_for_loss = (y_pred + torch.poisson(y_pred.detach() * 1e5) / 1e5) / 2
                    else:
                        y_pred_for_loss = y_pred

                    if self.config.s3im and self.current_projection > self.config.warmup.steps:
                        loss = 0
                        patch_size = min(math.floor(math.sqrt(self.projector.total_detector_pixels)), math.floor(math.sqrt(self.batch_size)),)
                        self.fabric.log_dict({"patch_size": patch_size}, step=self.step)
                        loss += self.loss_fn(y_pred_for_loss, y_target)
                        loss += self.s3im_loss(y_pred_for_loss, y_target, patch_size=patch_size)
                    else:
                        loss = self.loss_fn(y_pred_for_loss, y_target)

                    self.fabric.backward(loss)

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
            torch.cuda.empty_cache()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        mem_info = nvmlDeviceGetMemoryInfo(h)
        current_gb = mem_info.used / 1024**3
        print(f"[GPU] Peak allocated: {peak_gb:.2f} GB | Current used: {current_gb:.2f} GB / {total_gb:.1f} GB")
        self.evaluate()
        self.save_model(last=True)
