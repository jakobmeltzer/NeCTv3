from __future__ import annotations
import copy
import math
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import torch
import yaml
from dacite import from_dict
from loguru import logger
from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

import nect.sampling.geometry


@dataclass
class Geometry:
    nDetector: list[int] | tuple[int, int]
    dDetector: list[float] | tuple[float, float]
    nVoxel: list[int] | tuple[int, int, int]
    dVoxel: list[float] | tuple[float, float, float]
    offOrigin: list[float] | tuple[float, float, float]
    offDetector: list[float] | tuple[float, float]
    mode: str
    COR: float
    angles: list[float] | np.ndarray | torch.Tensor
    radians: bool
    timesteps: Optional[list[float] | np.ndarray | torch.Tensor] = None
    radius: Optional[float] = None
    height: Optional[float] = None
    remove_top: Optional[float] = None
    remove_bottom: Optional[float] = None
    flip: Optional[bool] = None
    rotDetector: Optional[list[float] | tuple[float, float, float]] = None
    invert_angles: Optional[bool] = None

    @property
    def num_angles(self) -> int:
        return len(self.angles)

    @property
    def sDetector(self) -> tuple[float, float]:
        sDetector = tuple([n * d for n, d in zip(self.nDetector, self.dDetector)])
        if len(sDetector) != 2:
            raise ValueError(f"Detector size {sDetector} is not valid")
        return sDetector

    @property
    def sVoxel(self) -> tuple[float, float, float]:
        sVoxel = tuple([n * d for n, d in zip(self.nVoxel, self.dVoxel)])
        if len(sVoxel) != 3:
            raise ValueError(f"Voxel size {sVoxel} is not valid")
        return sVoxel


@dataclass
class GeometryCone(Geometry):
    DSD: float = 0.0
    DSO: float = 0.0

    def __post_init__(self):
        if self.mode != "cone":
            raise ValueError(f"Geometry mode {self.mode} is not cone")
        if self.DSD <= 0:
            raise ValueError("DSD must be greater than 0")
        if self.DSO <= 0:
            raise ValueError("DSO must be greater than 0")


@dataclass
class HashEncoderConfig:
    otype: str
    n_levels: int
    n_features_per_level: int
    log2_hashmap_size: int
    base_resolution: int
    max_resolution_factor: float
    nDetector: Optional[list[int] | tuple[int, int]]
    nVoxel: Optional[list[int] | tuple[int, int, int]]
    sample_outside: Optional[int]
    max_resolution_type: Optional[Literal["nVoxel"] | Literal["nDetector"]] = "nDetector"
    
    def get_encoder_config(self) -> dict:
        return {
            "otype": self.otype,
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
        }
    
    def get_encoder_config_2D(self) -> dict:
        if self.log2_hashmap_size > 20:
            hashmax_2d = 20
        else:
            hashmax_2d = self.log2_hashmap_size

        return {
            "otype": self.otype,
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": hashmax_2d,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
        }

    def __str__(self):
        return f"{self.n_levels}_{self.n_features_per_level}_{self.log2_hashmap_size}_{self.base_resolution}_{self.max_resolution_factor}"

    @property
    def per_level_scale(self) -> float:
        if self.max_resolution_type == "nDetector":
            if self.nDetector is None:
                raise ValueError("nDetector is not provided")
            size = max(self.nDetector)
        elif self.max_resolution_type == "nVoxel":
            if self.nVoxel is None:
                raise ValueError("nVoxel is not provided")
            nVoxel = self.nVoxel
            if self.sample_outside != 0:
                nVoxel = [nVoxel[0], nVoxel[1] + 2*self.sample_outside, nVoxel[2] + 2 * self.sample_outside]
            size = max(nVoxel)
        return (self.max_resolution_factor * size / self.base_resolution) ** (1 / (self.n_levels - 1))


@dataclass
class DenseGridEncoderConfig:
    """2D multiresolution dense grid encoder (no hashing, no collisions).

    Generates a TCNN ``Grid`` encoding with ``type: Dense``.  For 2D inputs
    the grid is small enough that a dense layout is feasible; this eliminates
    the hash collisions that degrade ``HashGrid`` for pairwise encoders.

    ``per_level_scale`` is specified directly (not derived from geometry) since
    the 2D dense grid size is controlled by ``base_resolution * per_level_scale^(level-1)``.
    """

    n_levels: int
    n_features_per_level: int
    base_resolution: int
    per_level_scale: float

    def get_encoder_config(self) -> dict:
        return {
            "otype": "Grid",
            "type": "Dense",
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
        }

    def get_encoder_config_2D(self) -> dict:
        return self.get_encoder_config()

    def __str__(self):
        return f"{self.n_levels}_{self.n_features_per_level}_{self.base_resolution}_{self.per_level_scale}"


@dataclass
class MLPNetConfig:
    otype: str
    n_hidden_layers: int
    n_neurons: int
    activation: str
    output_activation: str
    include_adaptive_skip: Optional[bool]
    include_identity: Optional[bool]

    def get_network_config(self) -> dict:
        return {
            "otype": self.otype,
            "n_hidden_layers": self.n_hidden_layers,
            "n_neurons": self.n_neurons,
            "activation": self.activation,
            "output_activation": self.output_activation,
        }

    def __str__(self):
        return f"{self.n_hidden_layers}_{self.n_neurons}"

@dataclass
class KPlanesRegularizationConfig:
    space_lambda: float
    time_lambda: float
    time_type: str

@dataclass
class KPlanesEncoderConfig:
    grid_dimensions: int
    input_coordinate_dim: int
    output_coordinate_dim: int
    resolution: list[int]
    regularization: KPlanesRegularizationConfig

    def get_encoder_config(self) -> dict:
        return {
            "grid_dimensions": self.grid_dimensions,
            "input_coordinate_dim": self.input_coordinate_dim,
            "output_coordinate_dim": self.output_coordinate_dim,
            "resolution": self.resolution,
        }

    def __str__(self):
        return f"{self.grid_dimensions}_{self.input_coordinate_dim}_{self.output_coordinate_dim}_{self.resolution}"


@dataclass
class PirateNetConfig:
    n_modules: int
    alfa_init: float

    def get_network_config(self) -> dict:
        return {"n_modules": self.n_modules, "alfa_init": self.alfa_init}

    def __str__(self):
        return f"{self.n_modules}_{self.alfa_init}"


@dataclass
class TransformerDecoderConfig:
    """Config for QuadCubesTransformer's pure-PyTorch decoder.

    Fields:
        d_model  – projection dimension for each token (must be divisible by n_heads).
        n_heads  – number of self-attention heads.
        n_layers – number of TransformerEncoderLayer blocks.
        dropout  – dropout probability inside the transformer (default 0.0).
    """

    d_model: int
    n_heads: int
    n_layers: int
    dropout: float = 0.0

    def __str__(self):
        return f"transformer_{self.d_model}_{self.n_heads}_{self.n_layers}"


@dataclass
class UNetDecoderConfig:
    """Config for QuadCubesUNet's pure-PyTorch decoder.

    The multi-resolution hash grid is split into three scale groups
    (coarse / medium / fine), processed with a U-Net-style down+up path
    and skip connections.

    Fields:
        hidden_dims   – list of exactly 3 ints [d1, d2, d3]:
                        d1 = coarse-scale hidden width
                        d2 = medium-scale hidden width
                        d3 = bottleneck (fine-scale) width
        levels_coarse – number of hash grid levels assigned to the coarse group.
        levels_medium – number of hash grid levels assigned to the medium group.
                        The remainder (n_levels - levels_coarse - levels_medium)
                        forms the fine group.
        dropout       – reserved for future use (currently unused).
    """

    hidden_dims: list
    levels_coarse: int = 8
    levels_medium: int = 8
    dropout: float = 0.0

    def __str__(self):
        return f"unet_{'x'.join(map(str, self.hidden_dims))}"


@dataclass
class DownsamplingDetector:
    start: int
    end: int
    update_interval: int

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.update_interval))


@dataclass
class PointsPerRay:
    start: int
    end: int | Literal["auto"] | str
    update_interval: int | Literal["auto"] | str
    linear: Optional[bool]


@dataclass
class Evaluation:
    gt_path: str
    gt_path_mode: str
    evaluate_interval: float


@dataclass
class Warmup:
    steps: int
    lr0: float
    otype: Optional[str]


@dataclass
class LRScheduler:
    otype: str
    lrf: float | Literal["auto"]


@dataclass
class Optimizer:
    otype: str
    weight_decay: float
    beta1: float
    beta2: float = 0.999


@dataclass
class Crop:
    top: float
    bottom: float
    left_right: float

    def __hash__(self) -> int:
        return hash((self.top, self.bottom, self.left_right))


@dataclass
class Config:
    save_mode: Optional[str]
    uniform_ray_spacing: bool
    batch_per_proj: str | int
    add_poisson: bool
    points_per_batch: int | Literal["auto"]
    reconstruction_mode: str
    geometry: Geometry | GeometryCone
    img_path: str | pathlib.Path
    sparse_view: list[int] | None
    channel_order: str | None
    evaluation: Evaluation | None
    warmup: Warmup
    lr_scheduler: LRScheduler
    optimizer: Optimizer
    loss: str
    base_lr: float
    clip_grad_value: float | None
    epochs: int | Literal["auto"] | str
    plot_type: str | None
    image_interval: float
    checkpoint_interval: float
    points_per_ray: PointsPerRay
    s3im: bool
    model: str
    encoder: HashEncoderConfig | KPlanesEncoderConfig | DenseGridEncoderConfig
    net: MLPNetConfig | PirateNetConfig | TransformerDecoderConfig | UNetDecoderConfig
    concat: Optional[bool]
    encoder_2d: Optional[DenseGridEncoderConfig] = None
    temporal_encoder: Optional[HashEncoderConfig] = None
    reconstruction_mode: str
    save_volume: bool = False
    downsampling_detector: DownsamplingDetector = DownsamplingDetector(1, 1, 1)
    crop: Crop = Crop(0, 0, 0)
    use_prior: bool = False
    lr: float | None = None
    w0_steps: int | None = None
    w0_lr_multi: float | None = None 
    checkpoint_prior: Optional[str] = None
    checkpoint_epoch: Optional[int] = None 
    damp_multi: Optional[list[float]] = None 
    tv: float = 0.0
    tv_temporal: float = 0.0
    tv_spatial: float = 0.0
    n_levels_temporal: Optional[int] = None
    sample_outside: int = 0
    accumulation_steps: int | None = None
    continous_scanning: bool = False
    num_workers: int = 0
    use_extended_forward_model: bool = False
    beta_initial: float = 0.0

    @property
    def mode(self) -> str:
        if self.model in ["kplanes", "hash_grid", "tricubes"]:
            return "static"
        return "dynamic"
    
    def __post_init__(self):
        if isinstance(self.encoder, HashEncoderConfig):
            self.encoder.nDetector = self.geometry.nDetector
            self.encoder.nVoxel = self.geometry.nVoxel
            self.encoder.sample_outside = self.sample_outside
        if isinstance(self.temporal_encoder, HashEncoderConfig):
            self.temporal_encoder.nDetector = self.geometry.nDetector
            self.temporal_encoder.nVoxel = self.geometry.nVoxel
            self.temporal_encoder.sample_outside = self.sample_outside

    def get_dm(self) -> list[float]:
        return self.damp_multi
    
    def get_w0(self):
        return self.w0_steps, self.w0_lr_multi

    def get_str(self) -> str:
        return f"{self.model}_{str(self.encoder)}_{str(self.net)}_{self.loss}"

    def get_model(self) -> torch.nn.Module:
        """
        Get the model from the configuration.

        Returns:
            nn.Module: Model.
        """
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        memory_info = nvmlDeviceGetMemoryInfo(h)
        free_memory = int(memory_info.free)

        model = self.model
        byte_size = 4
        if model == "kplanes":
            from nect.network import KPlanes

            if isinstance(self.encoder, KPlanesEncoderConfig) and isinstance(self.net, MLPNetConfig):
                # memory_per_point = nodes_interpolation * byte_size (unknown) * self.encoder.n_levels (unknown) * num_encoders
                memory_per_point = 4 * 2 * 4 * 4 * self.encoder.output_coordinate_dim
                model = KPlanes(encoding_config=self.encoder, network_config=self.net)
            else:
                raise ValueError(f"Encoder and network configuration for model type {model} is not valid")
            
        elif model == "sexcubes_densegrid_transformer":
            if not isinstance(self.encoder, DenseGridEncoderConfig):
                raise ValueError(f"Encoder configuration for model type {model} must be DenseGridEncoderConfig")
            if not isinstance(self.net, TransformerDecoderConfig):
                raise ValueError(f"net must be TransformerDecoderConfig for {model}")
            from nect.network import SexCubesTransformer
            memory_per_point = 4 * 4 * self.encoder.n_levels * 6  # 4 nodes (bilinear), 4 bytes, 6 encoders
            model = SexCubesTransformer(
                encoding_config=self.encoder,
                decoder_config=self.net,
            )

        elif model in ["quadcubes_transformer", "quadcubes_unet", "sexcubes_transformer", "sexcubes_unet"]:
            if not isinstance(self.encoder, HashEncoderConfig):
                raise ValueError(f"Encoder configuration for model type {model} must be HashEncoderConfig")

            if model == "quadcubes_transformer":
                if not isinstance(self.net, TransformerDecoderConfig):
                    raise ValueError(f"net must be TransformerDecoderConfig for {model}")
                from nect.network import QuadCubesTransformer
                memory_per_point = 8 * byte_size * self.encoder.n_levels * 4
                model = QuadCubesTransformer(
                    encoding_config=self.encoder,
                    decoder_config=self.net,
                )

            elif model == "quadcubes_unet":
                if not isinstance(self.net, UNetDecoderConfig):
                    raise ValueError(f"net must be UNetDecoderConfig for {model}")
                from nect.network import QuadCubesUNet
                memory_per_point = 8 * byte_size * self.encoder.n_levels * 4
                model = QuadCubesUNet(
                    encoding_config=self.encoder,
                    decoder_config=self.net,
                )

            elif model == "sexcubes_transformer":
                if not isinstance(self.net, TransformerDecoderConfig):
                    raise ValueError(f"net must be TransformerDecoderConfig for {model}")
                from nect.network import SexCubesTransformer
                memory_per_point = 4 * byte_size * self.encoder.n_levels * 6  # 2D grids: 4 nodes
                model = SexCubesTransformer(
                    encoding_config=self.encoder,
                    decoder_config=self.net,
                )

            elif model == "sexcubes_unet":
                if not isinstance(self.net, UNetDecoderConfig):
                    raise ValueError(f"net must be UNetDecoderConfig for {model}")
                from nect.network import SexCubesUNet
                memory_per_point = 4 * byte_size * self.encoder.n_levels * 6
                model = SexCubesUNet(
                    encoding_config=self.encoder,
                    decoder_config=self.net,
                )

        elif model == "mixedcubes":
            if not isinstance(self.encoder, HashEncoderConfig):
                raise ValueError("mixedcubes: encoder must be HashEncoderConfig (3D spatial)")
            if not isinstance(self.encoder_2d, DenseGridEncoderConfig):
                raise ValueError("mixedcubes: encoder_2d must be DenseGridEncoderConfig (2D temporal)")
            if not isinstance(self.net, MLPNetConfig):
                raise ValueError("mixedcubes: net must be MLPNetConfig")
            from nect.network import MixedCubes
            memory_per_point = 4 * 4 * self.encoder_2d.n_levels * 3 + 8 * 4 * self.encoder.n_levels
            model = MixedCubes(
                encoding_config=self.encoder,
                encoding_config_2d=self.encoder_2d,
                network_config=self.net,
            )

        elif model == "mixedcubes_kplanes":
            if not isinstance(self.encoder, HashEncoderConfig):
                raise ValueError("mixedcubes_kplanes: encoder must be HashEncoderConfig (3D spatial)")
            if not isinstance(self.encoder_2d, DenseGridEncoderConfig):
                raise ValueError("mixedcubes_kplanes: encoder_2d must be DenseGridEncoderConfig (2D temporal)")
            if not isinstance(self.net, MLPNetConfig):
                raise ValueError("mixedcubes_kplanes: net must be MLPNetConfig")
            from nect.network import MixedCubesKPlanes
            memory_per_point = 4 * 4 * self.encoder_2d.n_levels * 3 + 8 * 4 * self.encoder.n_levels
            model = MixedCubesKPlanes(
                encoding_config=self.encoder,
                encoding_config_2d=self.encoder_2d,
                network_config=self.net,
            )

        elif model == "splitquadcubes":
            from nect.network import SplitQuadCubes

            if not (isinstance(self.encoder, HashEncoderConfig) and isinstance(self.net, MLPNetConfig)):
                raise ValueError("splitquadcubes: encoder must be HashEncoderConfig (spatial) and net must be MLPNetConfig")
            if not isinstance(self.temporal_encoder, HashEncoderConfig):
                raise ValueError("splitquadcubes: temporal_encoder must be HashEncoderConfig")

            spatial_features = self.encoder.n_levels * self.encoder.n_features_per_level
            temporal_features = self.temporal_encoder.n_levels * self.temporal_encoder.n_features_per_level
            memory_per_point = 8 * byte_size * (self.encoder.n_levels + 3 * self.temporal_encoder.n_levels)

            model = SplitQuadCubes(
                spatial_encoding_config=self.encoder,
                temporal_encoding_config=self.temporal_encoder,
                network_config=self.net,
            )

        elif model in ["hash_grid", "double_hash_grid", "quadcubes", "hypercubes", "tricubes", "sexcubes", "sexcubes_kplanes", "singlecube", "combinedcubes", "combinedcubes_kplanes"]:
            if not (isinstance(self.encoder, HashEncoderConfig) and isinstance(self.net, MLPNetConfig)):
                raise ValueError(f"Encoder and network configuration for model type {model} is not valid")
            
            if model == "hash_grid":
                from nect.network import HashGrid

                # memory_per_point = nodes_interpolation * byte_size * self.encoder.n_levels
                memory_per_point = 8 * byte_size * self.encoder.n_levels
                model = HashGrid(
                    encoding_config=self.encoder, 
                    network_config=self.net
                )

            elif model == "double_hash_grid":
                from nect.network import DoubleHashGrid

                # memory_per_point = nodes_interpolation * byte_size * self.encoder.n_levels
                memory_per_point = (8 + 16) * byte_size * self.encoder.n_levels
                model = DoubleHashGrid(
                    encoding_config=self.encoder,
                    network_config=self.net,
                )

            elif model == "quadcubes":
                from nect.network import QuadCubes

                # memory_per_point = nodes_interpolation * byte_size * self.encoder.n_levels * num_encoders
                memory_per_point = 8 * byte_size * self.encoder.n_levels * 4

                model = QuadCubes(
                    encoding_config=self.encoder,
                    network_config=self.net,
                    prior=self.use_prior,
                    concat=self.concat if self.concat is not None else True,
                )

            elif model == "tricubes":
                from nect.network import TriCubes

                # memory_per_point = nodes_interpolation * byte_size * self.encoder.n_levels * num_encoders
                memory_per_point = 8 * byte_size * self.encoder.n_levels * 3

                model = TriCubes(
                    encoding_config=self.encoder,
                    network_config=self.net
                )
            
            elif model == "sexcubes":
                from nect.network import SexCubes

                # memory_per_point = nodes_interpolation * byte_size * self.encoder.n_levels * num_encoders
                memory_per_point = 8 * byte_size * self.encoder.n_levels * 6

                model = SexCubes(
                    encoding_config=self.encoder,
                    network_config=self.net,
                )

            elif model == "sexcubes_kplanes":
                from nect.network import SexCubesKPlanes

                memory_per_point = 8 * byte_size * self.encoder.n_levels * 6

                model = SexCubesKPlanes(
                    encoding_config=self.encoder,
                    network_config=self.net,
                )

            elif model == "singlecube":
                from nect.network import SingleCube

                # memory_per_point = nodes_interpolation * byte_size * self.encoder.n_levels * num_encoders
                memory_per_point = 8 * byte_size * self.encoder.n_levels * 1

                model = SingleCube(
                    encoding_config=self.encoder,
                    network_config=self.net,
                )
            
            elif model == "combinedcubes":
                from nect.network import CombinedCubes

                temporal_levels = self.n_levels_temporal if self.n_levels_temporal is not None else self.encoder.n_levels
                memory_per_point = 8 * byte_size * (3 * temporal_levels + self.encoder.n_levels)

                model = CombinedCubes(
                    encoding_config=self.encoder,
                    network_config=self.net,
                    n_levels_temporal=self.n_levels_temporal,
                )

            elif model == "combinedcubes_kplanes":
                from nect.network import CombinedCubesKPlanes

                memory_per_point = 8 * byte_size * self.encoder.n_levels * 4

                model = CombinedCubesKPlanes(
                    encoding_config=self.encoder,
                    network_config=self.net,
                )

            elif model == "hypercubes":
                from nect.network import HyperCubes

                # memory_per_point = nodes_interpolation * byte_size * self.encoder.n_levels * num_encoders
                memory_per_point = 8 * byte_size * self.encoder.n_levels * 5

                model = HyperCubes(
                    encoding_config=self.encoder,
                    network_config=self.net,
                )
            else:
                raise ValueError(f"Model type {model} is not supported")
            
        else:
            raise ValueError(f"Model type {model} is not supported")
        # compute size of model
        # number of elements in the model
        memory_per_point *= 1.5  # buffer
        if isinstance(self.net, MLPNetConfig):
            memory_per_point += self.net.n_hidden_layers * self.net.n_neurons * 2 * 1.5  # 1.5 for buffer

        elif isinstance(self.encoder, HashEncoderConfig) and isinstance(self.net, PirateNetConfig):
            memory_per_point += (
                self.net.n_modules
                * self.encoder.n_features_per_level
                * self.encoder.n_levels
                * (2 if self.mode == "dynamic" else 1)
                * byte_size
            )

        model_and_optimizer_size = sum(p.numel() for p in model.parameters()) * 4 * 3
        if torch.cuda.device_count() > 1:
            model_and_optimizer_size *= 2  # ddp stores copy

        model_and_optimizer_size += 1024**3 * 3 + max(1024**3 * 3, model_and_optimizer_size * 0.3)  # buffer
        if self.points_per_batch == "auto":
            # memory_per_point = memory_per_point * 2 # buffer as we need for some calculation during backprop
            if self.uniform_ray_spacing:
                avg_memory_per_point = memory_per_point / 4 * 3
            else:
                avg_memory_per_point = memory_per_point

            self.points_per_batch = int((free_memory - model_and_optimizer_size) / avg_memory_per_point)
            self.points_per_batch = min(5_000_000, self.points_per_batch)
            logger.info(f"Setting points_per_batch to {self.points_per_batch:_}")
            if self.points_per_batch < 1:
                raise ValueError("Not enough memory to store even one point")
            
        return model

    def get_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        if self.points_per_batch == "auto":
            raise ValueError("`batch_per_proj` should already have been calculated as `get_model` should have been called before `get_optimizer`")
        
        if self.lr is None:
            self.lr = self.base_lr * math.sqrt((self.points_per_batch * torch.cuda.device_count()) / 1_000_000)

        if "adam".casefold() in self.optimizer.otype.casefold():
            if "adam".casefold() == self.optimizer.otype.casefold():
                adam_optim = torch.optim.Adam
            elif "nadam".casefold() == self.optimizer.otype.casefold():
                adam_optim = torch.optim.NAdam
            elif "adamw".casefold() == self.optimizer.otype.casefold():
                adam_optim = torch.optim.AdamW
            elif "radam".casefold() == self.optimizer.otype.casefold():
                adam_optim = torch.optim.RAdam
            optim = adam_optim(
                model.parameters(),
                lr=self.lr,
                betas=(self.optimizer.beta1, self.optimizer.beta2),
                weight_decay=self.optimizer.weight_decay,)
            
        elif "lion".casefold() == self.optimizer.otype.casefold():
            from lion_pytorch import Lion

            optim = Lion(
                model.parameters(),
                lr=self.lr,
                betas=(self.optimizer.beta1, self.optimizer.beta2),
                weight_decay=self.optimizer.weight_decay,
                use_triton=True,)
            
        elif "sgd".casefold() == self.optimizer.otype.casefold():
            optim = torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=self.optimizer.beta1,
                weight_decay=self.optimizer.weight_decay,)
            
        else:
            NotImplementedError(f"Optimizer {self.optimizer.otype} is not supported")

        return optim

    def get_lr_schedulers(self, optim: torch.optim.Optimizer) -> tuple[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.LRScheduler,]:
        warmup_factor_exponential = (1 / self.warmup.lr0) ** (1 / self.warmup.steps)

        def warmup_lr_exponential(projections):
            # return projections / self.warmup_steps
            if projections < self.warmup.steps:
                return 1 / (warmup_factor_exponential ** (self.warmup.steps - projections))
            else:
                return 1
            
        warmup_factor_linear = (1 - self.warmup.lr0) / self.warmup.steps
        def warmup_lr_linear(projections):
            if projections < self.warmup.steps:
                return projections * warmup_factor_linear
            return 1
        
        if self.warmup.otype == "linear":
            warmup_lr = warmup_lr_linear
        else:
            warmup_lr = warmup_lr_exponential

        lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
        lr_scheduler_warmup_downsample = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda x: x / 500)
        if isinstance(self.epochs, str):
            raise ValueError("Epcohs has not been set yet, although it should have been.")
        # gamma = (1/100) ** (1/(config["epochs"] * len(geometry["angles"])))
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
        if self.lr_scheduler.lrf == "auto":
            self.lr_scheduler.lrf = max(1e-2, min(0.5, 1 / (self.epochs * 0.1)))

        if "exponential".casefold() == self.lr_scheduler.otype.casefold():
            gamma = self.lr_scheduler.lrf ** (1 / (self.epochs * len(self.geometry.angles)))
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

        elif "cosine".casefold() == self.lr_scheduler.otype.casefold():
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=self.epochs * len(self.geometry.angles) // torch.cuda.device_count(),
                eta_min=self.lr * self.lr_scheduler.lrf,
            )
        return lr_scheduler_warmup, lr_scheduler, lr_scheduler_warmup_downsample

    def get_loss_fn(self):
        if "l1".casefold() in self.loss.casefold():
            def l1_loss(pred, target, *args, **kwargs):
                return torch.nn.functional.l1_loss(pred, target)
            return l1_loss
        
        elif "l2".casefold() in self.loss.casefold():
            def l2_loss(pred, target, *args, **kwargs):
                return torch.nn.functional.mse_loss(pred, target)
            return l2_loss
        
        elif "l1+l2".casefold() in self.loss.casefold():
            def l1_l2_loss(pred, target, i, *args, **kwargs):
                if i % 2 == 0:
                    return torch.nn.functional.l1_loss(pred, target)
                return torch.nn.functional.mse_loss(pred, target)
            return l1_l2_loss
        
        else:
            raise NotImplementedError(f"Loss {self.loss} is not supported")

    def save(self, output_directory: str | pathlib.Path):
        output_directory = pathlib.Path(output_directory)
        model_dir = output_directory / "model"
        model_dir.mkdir(parents=True, exist_ok=True)   # <-- this fixes the error

        # Save config.yaml
        with open(model_dir / "config.yaml", "w") as f:
            config = asdict(self)
            geometry = config.pop("geometry")
            config["geometry"] = "SAME_FOLDER"
            config["img_path"] = str(config["img_path"])
            config["encoder"].pop("nDetector", None)
            config["encoder"].pop("nVoxel", None)
            config["mode"] = self.mode
            yaml.dump(config, f)

        # Save geometry.yaml
        with open(model_dir / "geometry.yaml", "w") as f:
            if isinstance(geometry["angles"], np.ndarray):
                geometry["angles"] = geometry["angles"].tolist()
            if isinstance(geometry["timesteps"], np.ndarray):
                geometry["timesteps"] = geometry["timesteps"].tolist()

            geometry["angles"] = list(geometry["angles"])
            if geometry["timesteps"] is not None:
                geometry["timesteps"] = list(geometry["timesteps"])

            geometry["nDetector"] = list(geometry["nDetector"])
            geometry["dDetector"] = list(geometry["dDetector"])
            geometry["nVoxel"] = list(geometry["nVoxel"])
            geometry["dVoxel"] = list(geometry["dVoxel"])
            geometry["offOrigin"] = list(geometry["offOrigin"])
            geometry["offDetector"] = list(geometry["offDetector"])
            geometry["rotDetector"] = list(geometry["rotDetector"])
            yaml.dump(geometry, f, default_flow_style=None)

    def get_s3im_loss(self):
        from torch_extra.nn.modules import SSIM

        ssim = SSIM(data_range=1.0, channel=1, spatial_dims=2, stride=5, win_size=5)

        def s3im_loss(pred, target, patch_size):
            loss = 0
            pred = pred.float()
            target = target.float()
            if patch_size >= 5:
                for i in range(5):
                    suffle_idx = torch.randperm(target.size(0))
                    y_pred2 = pred[suffle_idx][: patch_size**2]
                    y2 = target[suffle_idx][: patch_size**2]
                    ssim_value = 1 - ssim(
                        y_pred2.reshape(-1, patch_size, patch_size).unsqueeze(1),
                        y2.reshape(-1, patch_size, patch_size).unsqueeze(1),)
                    loss += ssim_value / 5
                loss = loss * 3
            return loss

        return s3im_loss


def load_config(config) -> dict:
    with open(config, "r") as f:
        return yaml.safe_load(f)


cfg_paths: dict = {
    "static": {
        "hash_grid": pathlib.Path(__file__).parent / "cfg/static/hash_grid.yaml",
        "kplanes": pathlib.Path(__file__).parent / "cfg/static/kplanes.yaml",
        "tricubes": pathlib.Path(__file__).parent / "cfg/static/tricubes.yaml",
    },
    "dynamic": {
        "kplanes_dynamic": pathlib.Path(__file__).parent / "cfg/dynamic/kplanes_dynamic.yaml",
        "double_hash_grid": pathlib.Path(__file__).parent / "cfg/dynamic/double_hash_grid.yaml",
        "quadcubes": pathlib.Path(__file__).parent / "cfg/dynamic/quadcubes.yaml",
        "hypercubes": pathlib.Path(__file__).parent / "cfg/dynamic/hypercubes.yaml",
        "sexcubes": pathlib.Path(__file__).parent / "cfg/dynamic/sexcubes.yaml",
        "sexcubes_kplanes": pathlib.Path(__file__).parent / "cfg/dynamic/sexcubes_kplanes.yaml",
        "singlecube": pathlib.Path(__file__).parent / "cfg/dynamic/singlecube.yaml",
        "combinedcubes": pathlib.Path(__file__).parent / "cfg/dynamic/combinedcubes.yaml",
        "quadcubes_transformer": pathlib.Path(__file__).parent / "cfg/dynamic/quadcubes_transformer.yaml",
        "quadcubes_unet": pathlib.Path(__file__).parent / "cfg/dynamic/quadcubes_unet.yaml",
        "sexcubes_transformer": pathlib.Path(__file__).parent / "cfg/dynamic/sexcubes_transformer.yaml",
        "sexcubes_unet": pathlib.Path(__file__).parent / "cfg/dynamic/sexcubes_unet.yaml",
        "sexcubes_densegrid_transformer": pathlib.Path(__file__).parent / "cfg/dynamic/sexcubes_densegrid_transformer.yaml",
        "mixedcubes": pathlib.Path(__file__).parent / "cfg/dynamic/mixedcubes.yaml",
        "combinedcubes_kplanes": pathlib.Path(__file__).parent / "cfg/dynamic/combinedcubes_kplanes.yaml",
        "mixedcubes_kplanes": pathlib.Path(__file__).parent / "cfg/dynamic/mixedcubes_kplanes.yaml",
    },
}


def get_default_cfg() -> dict:
    """
    Load default configuration from file and return it as a dictionary.

    Returns:
        dict: Default configuration."""
    return load_config(pathlib.Path(__file__).parent / "cfg/default.yaml")


def get_static_cfg(name: str) -> dict:
    """
    Load static configuration from file and return it as a dictionary.

    Args:
        name (str): Model type.

    Returns:
        dict: Static default configuration for the model type.
    """
    default_cfg = get_default_cfg()
    static_cfg = load_config(cfg_paths["static"][name])
    default_cfg.update(static_cfg)
    default_cfg.update({"mode": "static"})
    return default_cfg


def get_dynamic_cfg(name: str) -> dict:
    """
    Load dynamic configuration from file and return it as a dictionary.

    Args:
        name (str): Model type.

    Returns:
        dict: Dynamic default configuration for the model type.
    """
    default_cfg = get_default_cfg()
    dynamic_cfg = load_config(cfg_paths["dynamic"][name])
    default_cfg.update(dynamic_cfg)
    default_cfg.update({"mode": "dynamic"})
    return default_cfg


def cfg_dict_to_dataclass(config: dict):
    geometry_cfg = load_config(config.get("geometry"))
    if geometry_cfg is None:
        raise ValueError("Geometry configuration is not provided")
    
    config.update({"geometry": geometry_cfg})
    cfg_sanity_check(config)
    cfg: Config = from_dict(data_class=Config, data=config)
    if cfg.geometry.mode == "cone":
        cfg.geometry = from_dict(data_class=GeometryCone, data=geometry_cfg)

    return cfg


def get_cfg(path: str | pathlib.Path, model: str | None = None, static: bool | None = None) -> Config:
    """
    Load configuration from file and return it as a dictionary.

    Args:
        path (str): Path to the configuration file.
        model (str, optional): Model type. Defaults to None.
        static (bool, optional): Static mode. Defaults to None.

    Returns:
        dict: Configuration.
    """
    cfg_specific = None
    if model is None or static is None:
        cfg_specific = load_config(path)
        if model is None:
            model = cfg_specific.get("model")
            if model is None:
                raise ValueError("Model type via parameter 'model' is not provided")
            
        if static is None:
            static_cfg = cfg_specific.get("mode")
            if static_cfg not in ["static", "dynamic"]:
                if model in ["kplanes", "hash_grid", "tricubes"]:
                    static_cfg = "static"
                else:
                    static_cfg = "dynamic"

            static = static_cfg == "static"

    if static:
        cfg = get_static_cfg(model)
    else:
        cfg = get_dynamic_cfg(model)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")
    
    if cfg_specific is None:
        cfg.update(load_config(path))
    else:
        cfg.update(cfg_specific)

    if cfg.get("geometry") == "SAME_FOLDER":
        cfg["geometry"] = os.path.join(os.path.dirname(path), "geometry.yaml")

    if not pathlib.Path(cfg["img_path"]).is_absolute():
        cfg["img_path"] = os.path.join(os.path.dirname(path), cfg["img_path"])

    geometry_cfg = load_config(cfg.get("geometry"))
    if geometry_cfg is None:
        raise ValueError("Geometry configuration is not provided")
    
    cfg.update({"geometry": geometry_cfg})
    cfg_sanity_check(cfg)
    if cfg.get("lr") is None:
        cfg["lr"] = None
    return setup_cfg(cfg)


def setup_cfg(cfg: dict) -> Config:
    """
    Setup configuration from a dict and return it as a dataclass.

    Args:
        cfg (dict): Configuration.

    Returns:
        Config: Configuration as a dataclass.
    """
    config = from_dict(data_class=Config, data=cfg)
    if config.geometry.mode == "cone":
        config.geometry = from_dict(data_class=GeometryCone, data=cfg["geometry"])

    return config


def cfg_sanity_check(cfg: dict):
    """
    Check if the configuration is valid.

    Args:
        cfg (dict): Configuration.

    """
    mode = cfg.get("mode")
    if mode not in ["static", "dynamic"]:
        raise ValueError(f"Mode {mode} is not valid, must be either 'static' or 'dynamic'")
    
    in_func = lambda x, y: x.casefold() in [y_el.casefold() for y_el in y]
    supported_activation_functions = ["none", "relu", "sigmoid", "leakyrelu", "exponential", "tanh", "sine", "squareplus", "softplus",]

    in_list = "in the list"
    gt_eq = "greater or equal to"
    gt = "greater than"
    lt_eq = "less or equal to"
    eq = "equal to"
    len_eq = "length equal to"
    correct_type = "correct type"
    hash_encoder = {
        "otype": (str, [(str.__eq__, "HashGrid", eq)]),
        "n_levels": [(int, [(int.__ge__, 1)], gt_eq)],
        "n_features_per_level": (int, [(int.__ge__, 1, gt_eq)]),
        "log2_hashmap_size": (int, [(int.__ge__, 1, gt_eq)]),
        "base_resolution": (int, [(int.__ge__, 1, gt_eq)]),
        "max_resolution_factor": (float, [(float.__gt__, 0.0, gt)]),
        "max_resolution_type": (Optional[str], [(in_func, ["nVoxel", "nDetector"], in_list)])}
    
    mlp_net = {
        "otype": (str, [(in_func, ["FullyFusedMLP", "CutlassMLP"], in_list)]),
        "n_hidden_layers": (int, [(int.__ge__, 1, gt_eq)]),
        "n_neurons": (int, [(int.__ge__, 1, gt_eq)]),
        "activation": (str, [(in_func, supported_activation_functions, in_list)]),
        "output_activation": (str, [(in_func, supported_activation_functions, in_list)]),
        "include_adaptive_skip": (Optional[bool], []),
        "include_identity": (Optional[bool], []),}

    sanity = {
        "hash_grid": {"encoder": hash_encoder, "net": mlp_net},
        "kplanes": {
            "encoder": {
                "grid_dimensions": (int, [(int.__eq__, 2, eq)]),
                "input_coordinate_dim": (int, [(lambda x, y: x == y, 3 if mode == "static" else 4, eq)],),
                "output_coordinate_dim": (int, [(int.__ge__, 1, gt_eq)]),
                "resolution": (list, [(lambda x, y: len(x) == y, 3 if mode == "static" else 4, len_eq,), (lambda x, y: all(isinstance(i, y) for i in x), int, correct_type,),],),
                "regularization": {
                    "space_lambda": (float, [(float.__ge__, 0.0, gt_eq)]),
                    "time_lambda": (float, [(float.__ge__, 0.0, gt_eq)]),
                    "time_type": (str, [(in_func, ["l1", "smoothnes"], in_list)],),
                },
            },
            "net": mlp_net,
        },
        "double_hash_grid": {"encoder": hash_encoder, "net": mlp_net},
        "quadcubes": {"encoder": hash_encoder, "net": mlp_net, "cat": (Optional[bool], []),},
        "tricubes": {"encoder": hash_encoder, "net": mlp_net, "cat": (Optional[bool], []),},
        "sexcubes": {"encoder": hash_encoder, "net": mlp_net, "cat": (Optional[bool], []),},
        "sexcubes_kplanes": {"encoder": hash_encoder, "net": mlp_net},
        "singlecube": {"encoder": hash_encoder, "net": mlp_net, "cat": (Optional[bool], []),},
        "combinedcubes": {"encoder": hash_encoder, "net": mlp_net, "cat": (Optional[bool], []),},
        "combinedcubes_kplanes": {"encoder": hash_encoder, "net": mlp_net},
        "hypercubes": {"encoder": hash_encoder, "net": mlp_net, "cat": (Optional[bool], []), },
        "quadcubes_transformer": {"encoder": hash_encoder},
        "quadcubes_unet": {"encoder": hash_encoder},
        "sexcubes_transformer": {"encoder": hash_encoder},
        "sexcubes_unet": {"encoder": hash_encoder},
        "sexcubes_densegrid_transformer": {"encoder": {
            "n_levels": (int, [(int.__ge__, 1, gt_eq)]),
            "n_features_per_level": (int, [(int.__ge__, 1, gt_eq)]),
            "base_resolution": (int, [(int.__ge__, 1, gt_eq)]),
            "per_level_scale": (float, [(float.__gt__, 1.0, gt)]),
        }},
        "mixedcubes": {"encoder": hash_encoder},
        "mixedcubes_kplanes": {"encoder": hash_encoder},}
    
    sanity["kplanes_dynamic"] = sanity["kplanes"]
    sanity_all = {
        "image_interval": (float, []),
        "checkpoint_interval": (float, []),
        "epochs": (str, [(lambda x, y: x == y or (len(x.split("x")) == 2 and float(x.split("x")[0]) > 0) or int(x) > 0, "auto", "not a int or 'auto' or '<float>x'",)],),
        "loss": (str, [(in_func, ["L1", "L2", "L1+L2"], in_list)]),
        "optimizer": {
            "otype": (str, [(in_func, ["Adam", "NAdam", "RAdam", "Lion", "SGD"], in_list)],),
            "weight_decay": (float, [(float.__ge__, 0.0, gt_eq)]),
            "beta1": (float, [(float.__ge__, 0.0, gt_eq), (float.__le__, 1.0, lt_eq)]),
            "beta2": (Optional[float], [(float.__ge__, 0.0, gt_eq), (float.__le__, 1.0, lt_eq)],),
        },
        "lr_scheduler": {
            "otype": (str, [(in_func, ["Exponential", "Cosine"], in_list)]),
            "lrf": (str, [(lambda x, y: x == y or (float(x) > 0 and float(x) <= 1), "auto", "not a float or 'auto'",)],),
        },
        "warmup": {
            "steps": (int, [(int.__ge__, 0, gt_eq)]),
            "lr0": (float, [(float.__gt__, 0.0, gt_eq)]),
            "otype": (Optional[str], [(in_func, ["Linear", "Exponential"], in_list)])
        },
        "s3im": (bool, []),
        "points_per_ray": {
            "start": (int, [(int.__ge__, 1, gt_eq)]),
            "end": (str, [(lambda x, y: x == y or (len(x.split("x")) == 2 and float(x.split("x")[0]) > 0) or int(x) > 0, "auto", "not a int or 'auto' or '<float>x'",)],),
            "update_interval": (str, [(lambda x, y: x == y or (len(x.split("x")) == 2 and float(x.split("x")[0]) > 0) or int(x) > 0, "auto", "not a int or 'auto' or '<float>x'")],),
            "linear": (Optional[bool], []),
        },
        "batch_per_proj": (str,  [(lambda x, y: x == y or int(x) > 0, "all", "not a int or 'all'")],),
        "add_poisson": (bool, []),
        "points_per_batch": (str, [(lambda x, y: x == y or int(x) > 0, "auto", "not a int or 'auto'")],),
        "reconstruction_mode": (str, [(in_func, ["voxel", "cylindrical"], in_list)]),
        "img_path": (str, [], ),
        "uniform_ray_spacing": (bool, []),
        "geometry": {
            "nDetector": (list, [(lambda x, y: len(x) == y, 2, len_eq), (lambda x, y: all(isinstance(i, y) for i in x), int, correct_type),],),
            "dDetector": (list, [(lambda x, y: len(x) == y, 2, len_eq), (lambda x, y: all(isinstance(i, y) for i in x), float | int, correct_type,),],),
            "nVoxel": (list, [(lambda x, y: len(x) == y, 3, len_eq), (lambda x, y: all(isinstance(i, y) for i in x), int, correct_type),],),
            "dVoxel": (list, [(lambda x, y: len(x) == y, 3, len_eq), (lambda x, y: all(isinstance(i, y) for i in x), float | int, correct_type,),],),
            "offOrigin": (list, [(lambda x, y: len(x) == y, 3, len_eq), (lambda x, y: all(isinstance(i, y) for i in x), float | int, correct_type,),],),
            "offDetector": (list, [(lambda x, y: len(x) == y, 2, len_eq), (lambda x, y: all(isinstance(i, y) for i in x), float | int, correct_type,),],),
            "rotDetector": (Optional[list], [(lambda x, y: len(x) == y, 3, len_eq), (lambda x, y: all(isinstance(i, y) for i in x), float | int, correct_type,),],),
            "mode": (str, [(in_func, ["parallel", "cone"], in_list)]),
            "COR": (float, []),
            "angles": (list, [(lambda x, y: all(isinstance(i, y) for i in x), float | int, correct_type,)],),
            "radians": (bool, []),
            "timesteps": (Optional[list], []),
            "radius": (Optional[float], []),
            "height": (Optional[float], []),
            "remove_top": (Optional[float], []),
            "remove_bottom": (Optional[float], []),
            "flip": (Optional[bool], []),
            "DSD": (Optional[float], []),
            "DSO": (Optional[float], []),
            "invert_angles": (Optional[bool], []), },}
    
    sanity_optional = {
        "save_volume": (bool, []),
        "clip_grad_value": (float, [(float.__ge__, 0.0, gt_eq)]),
        "plot_type": (str, [(in_func, ["XY", "XZ", "YZ"])]),
        "sparse_view": (list, [(lambda x, y: len(x) == y, 2, len_eq), (lambda x, y: all(isinstance(i, y) for i in x), int, correct_type), (lambda x, y: x[0] < x[1], None, "first element is not less than the second",),],),
        "channel_order": (str, []),
        "downsampling_detector": {"start": (int, [(int.__ge__, 1, gt_eq)]), "end": (int, [(int.__ge__, 1, gt_eq)]), "update_interval": (int, [(int.__ge__, 1, gt_eq)]),},
        "evaluation": {
            "gt_path": (str, []), 
            "gt_path_mode": (str,[(lambda x, y: x in y or os.path.exists(x), ["SciVis", "PorousMedium"], "path does not exist",)],),
            "evaluate_interval": (float, [(float.__ge__, 1.0, gt_eq)]),
            },
        "crop": {
            "top": (float, [(float.__ge__, 0.0, gt_eq)]),
            "bottom": (float, [(float.__ge__, 0.0, gt_eq)]),
            "left_right": (float, [(float.__ge__, 0.0, gt_eq)]),
        },
        "use_prior": (bool, []),
        "checkpoint_prior": (Optional[str], []),
        "base_lr": (float, [(float.__gt__, 0.0, gt_eq)]),
        "lr": (float, [(float.__gt__, 0.0, gt_eq)]),
        "tv": (float, [(float.__ge__, 0.0, gt_eq)]),
        "tv_temporal": (float, [(float.__ge__, 0.0, gt_eq)]),
        "n_levels_temporal": (int, [(int.__ge__, 1, gt_eq)]),
        "sample_outside": (int, [(int.__ge__, 0, gt_eq)]),
        "accumulation_steps": (Optional[int], [(int.__ge__, 1, gt_eq)]),
        "continous_scanning": (bool, []),
        "use_extended_forward_model": (bool, []),
        "beta_initial": (float, []),
        "num_workers": (int, [(int.__ge__, 0, gt_eq)]),}
    
    model = cfg.get("model")
    if model is None:
        raise ValueError("Model type is not provided")

    if model not in sanity.keys():
        raise ValueError(f"Model type {model} is not supported, must be one of {list(sanity.keys())}")
    
    check_cfg(sanity[model], cfg)
    check_cfg(sanity_all, cfg)
    check_cfg(sanity_optional, cfg, optional=True)


def check_cfg(sanity: dict, cfg: dict, nested_key: list = [], optional: bool = False):
    """
    Check if the configuration is valid.

    Args:
        sanity (dict): Configuration sanity check.
        cfg (dict): Configuration.

    Raises:
        ValueError: If the configuration is not valid.
    """
    for key, value in sanity.items():
        cfg_value = cfg.get(key)
        nested_key_ = copy.deepcopy(nested_key)
        nested_key_.append(key)
        if cfg_value is None:  # check if the value is provided
            if optional or isinstance(None, value[0]):  # check if the value is optional
                continue
            raise ValueError(f"Key {key} is not in the configuration at key {nested_key_}")
        
        if isinstance(value, dict):
            if not isinstance(cfg_value, dict):
                raise ValueError(f"Value {cfg_value} is not a dictionary, but is supposed to be, at key {nested_key_}")
            check_cfg(value, cfg_value, nested_key_)

        elif isinstance(value, tuple):
            try:
                cast = value[0]
                if isinstance(None, value[0]):
                    cast = value[0].__args__[0]  # get the type of the optional value
                typed_cfg_value = cast(cfg_value)
                if cast in [float, int]:
                    assert (
                        typed_cfg_value == cfg_value
                    ), f"Value {cfg_value} is not of type {value[0]} at key {nested_key_}"
            except Exception as e:
                raise ValueError(f"Value {cfg_value} is not of type {value[0]} at key {nested_key_}") from e
            
            for check in value[1]:
                if not check[0](typed_cfg_value, check[1]):
                    raise ValueError(f"Value {cfg_value} does not satisfy the condition {check[2], check[1]} at key {nested_key_}")


def get_config(
    geometry: nect.sampling.geometry.Geometry,
    img_path: str | pathlib.Path,
    model: str | None = None,
    mode: str = "static",
    channel_order: str | None = None,
) -> Config:
    """
    Get configuration.

    Args:
        geometry (nect.sampling.geometry.Geometry): Geometry.
        img_path (str | pathlib.Path): Image path.
        model (str, optional): Model type. Defaults to None.
        mode (str, optional): Mode. Defaults to "static".
        channel_order (str, optional): Channel order. Defaults to None.

    Returns:
        Config. Configuration as a dataclass.
    """
    if mode not in ["static", "dynamic"]:
        raise ValueError(f"Mode {mode} is not valid, must be either 'static' or 'dynamic")
    
    if model is None:
        model = "hash_grid" if mode == "static" else "quadcubes"

    if mode == "static":
        cfg = get_static_cfg(model)
    else:
        cfg = get_dynamic_cfg(model)

    cfg.update({"geometry": geometry.to_dict(), "img_path": img_path, "channel_order": channel_order, "model": model})
    return setup_cfg(cfg)


