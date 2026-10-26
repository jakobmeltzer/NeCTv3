"""
Demo 11: Dynamic reconstruction using QuadCubes with a large spatial encoder on the Bentheimer dataset.
Uses separate spatial (n_levels=24) and temporal (n_levels=23) hash grids for high-resolution spacetime encoding.
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
    exp_name="splitquadcubes_large_spatial",
    config_override={
        "model": "splitquadcubes",
        "epochs": "8x",
        "checkpoint_interval": 0,
        "image_interval": 0,
        "plot_type": "XZ",
        "base_lr": 0.0002,
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
        "temporal_encoder": {
            "otype": "HashGrid",
            "n_levels": 23,
            "n_features_per_level": 4,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "max_resolution_factor": 2,
        },
        "net": {
            "otype": "FullyFusedMLP",
            "activation": "LeakyReLU",
            "output_activation": "None",
            "n_neurons": 128,
            "n_hidden_layers": 4,
            "include_identity": False,
            "include_adaptive_skip": False,
        },
    },
)
