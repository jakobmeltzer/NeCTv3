#!/usr/bin/env python3
"""Compare NHW and WH interpretations of one Bentheimer raw projection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_PROJECTION_DIR = Path(
    "/cluster/work/jakobkm/NeCTv3/nect/data/Bentheimer/projections"
)
DETECTOR_HEIGHT = 1880
DETECTOR_WIDTH = 1496
EXPECTED_PROJECTIONS = 1400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load one Bentheimer .raw projection as both NHW and WH and "
            "compare the resulting detector arrays."
        )
    )
    parser.add_argument(
        "projection_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_PROJECTION_DIR,
        help=f"Directory containing projection .raw files (default: {DEFAULT_PROJECTION_DIR})",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Zero-based index of the sorted projection file to inspect (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    projection_dir = args.projection_dir.expanduser()

    if not projection_dir.is_dir():
        raise SystemExit(f"Projection directory does not exist: {projection_dir}")

    files = sorted(projection_dir.glob("*.raw"))
    if not files:
        raise SystemExit(f"No .raw files found in: {projection_dir}")
    if not 0 <= args.index < len(files):
        raise SystemExit(
            f"--index must be between 0 and {len(files) - 1}; got {args.index}"
        )

    projection_file = files[args.index]
    raw = np.fromfile(projection_file, dtype=np.float32)
    expected_values = DETECTOR_HEIGHT * DETECTOR_WIDTH

    print(f"Projection directory: {projection_dir}")
    print(f"Raw files found:     {len(files)}")
    print(f"Expected files:      {EXPECTED_PROJECTIONS}")
    print(f"Projection inspected:{projection_file}")
    print(f"Float32 values:      {raw.size}")
    print(f"Expected values:     {expected_values}")

    if len(files) != EXPECTED_PROJECTIONS:
        print("WARNING: projection count is not 1400")
    if raw.size != expected_values:
        raise SystemExit(
            f"Unexpected raw file size: expected {expected_values} float32 values, "
            f"found {raw.size}"
        )

    # This reproduces the loader's behavior for a directory of individual raw
    # files when channel_order is incorrectly set to NHW.
    as_nhw = raw.reshape(-1, DETECTOR_HEIGHT, DETECTOR_WIDTH)[0]

    # This reproduces the loader's behavior for the Bentheimer dataset's WH
    # storage order: reshape as (W, H), then transpose to the projector's (H, W).
    as_wh = raw.reshape(DETECTOR_WIDTH, DETECTOR_HEIGHT).T

    different_pixels = np.count_nonzero(as_nhw != as_wh)
    different_fraction = different_pixels / raw.size

    print()
    print(f"NHW final shape:     {as_nhw.shape}")
    print(f"WH final shape:      {as_wh.shape}")
    print(f"Same final shape:    {as_nhw.shape == as_wh.shape}")
    print(f"Arrays equal:        {np.array_equal(as_nhw, as_wh)}")
    print(f"Different pixels:    {different_pixels} / {raw.size}")
    print(f"Different fraction:  {different_fraction:.8%}")
    print()
    print("NHW first row:   ", as_nhw[0, :8])
    print("WH first row:    ", as_wh[0, :8])
    print("NHW first column:", as_nhw[:8, 0])
    print("WH first column: ", as_wh[:8, 0])

    if np.array_equal(as_nhw, as_wh):
        print("\nRESULT: NHW and WH produced identical arrays.")
    else:
        print(
            "\nRESULT: NHW and WH have the same detector shape but different "
            "pixel placement."
        )


if __name__ == "__main__":
    main()
