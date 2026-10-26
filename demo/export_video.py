from pathlib import Path
import nect

model_path = Path(
    "/cluster/work/jakobkm/NeCTv3/outputs/mixedcubes/mixedcubes_24_4_24_16_2_4_128_L1/2026-10-23T17-44-48/model"
)

nect.export_video(
    model_path,
    plot_slice="XZ",
    fps=10,
    difference=False,
    video_name="mixedcubes_final",
)