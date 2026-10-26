"""
Demo 10: Dynamic reconstruction using the MixedCubes architecture on the Bentheimer dataset.
MixedCubes combines 3D spatial hash grids with 2D temporal planes at mixed resolutions.
"""

from nect.download_demo_data import download_demo_data, get_demo_data_path
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
    exp_name="mixedcubes",
    config_override={
        "epochs": "8x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0005,
        "warmup": {
            "steps": 2000,
            "lr0": 0.001,
        },
        "encoder": {
            "otype": "HashGrid",
            "n_levels": 24,
            "n_features_per_level": 4,
            "log2_hashmap_size": 24,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "encoder_2d": {
            "n_levels": 12,
            "n_features_per_level": 4,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        },
        "net": {
            "otype": "FullyFusedMLP",
            "activation": "LeakyReLU",
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 4,
            "include_identity": False,
        },
        "tv_spatial": 1e-4,
    },
    enc_arc="mixedcubes",
)
