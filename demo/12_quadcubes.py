"""
Demo 12: Dynamic reconstruction using the default QuadCubes architecture on
the Bentheimer dataset.

This is a shorter comparison run for demo 10. It uses the default QuadCubes
encoder, network, learning-rate, and warmup configuration, and trains for 1x
instead of the 8x used by the MixedCubes demo.
"""

from nect.download_demo_data import download_demo_data, get_demo_data_path
import nect


download_demo_data("Bentheimer")
demo_dir = get_demo_data_path("Bentheimer")
geometry = nect.Geometry.from_yaml(demo_dir / "geometry.yaml")

reconstruction_path, output_directory = nect.reconstruct(
    geometry=geometry,
    projections=demo_dir / "projections",
    channel_order="WH",
    quality="high",
    mode="dynamic",
    exp_name="quadcubes_default_bentheimer",
    config_override={
        "epochs": "1x",
        "checkpoint_interval": 1800,
        "image_interval": 30,
        "plot_type": "XZ",
    },
    enc_arc="quadcubes",
)

print(f"Reconstruction model: {reconstruction_path}")
print(f"Run directory: {output_directory}")
