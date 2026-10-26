"""
Demo 09: Dynamic reconstruction using the CombinedCubes architecture on the Bentheimer dataset.
CombinedCubes couples a static spatial hash grid with a dynamic temporal grid.
"""

from nect.download_demo_data import download_demo_data, get_demo_data_path
from nect.config import MLPNetConfig
import nect

download_demo_data("Bentheimer")
demo_dir = get_demo_data_path("Bentheimer")
geometry = nect.Geometry.from_yaml(demo_dir / "geometry.yaml")

reconstruction_path, _ = nect.reconstruct(
    geometry=geometry,
    projections=demo_dir / "projections",
    channel_order="WH",
    quality="high",
    mode="dynamic",
    exp_name="combinedcubes",
    config_override={
        "epochs": "8x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0001,
        "warmup": {
            "steps": 1400 * 10,
            "lr0": 0.0001,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 24,
            "n_features_per_level": 4,
            "log2_hashmap_size": 24,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "net": MLPNetConfig(
            otype="FullyFusedMLP",
            activation="LeakyReLU",
            output_activation="None",
            n_neurons=128,
            n_hidden_layers=4,
            include_identity=False,
            include_adaptive_skip=False,
        ),
        "tv_spatial": 1e-4,
        "n_levels_temporal": 18,
    },
    enc_arc="combinedcubes",
)
