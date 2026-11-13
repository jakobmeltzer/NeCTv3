from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .porous_phantom import generate_porous_phantom


def _obtain_stage2_geometry(config: dict, output_dir: Path, stage2_dir: Path) -> tuple[np.ndarray, np.ndarray, str]:
    source_pore = stage2_dir / "ground_truth" / "pore_mask.npy"
    source_rock = stage2_dir / "ground_truth" / "rock_mask.npy"
    expected = tuple(config["volume"]["shape_zyx"])
    if source_pore.exists() and source_rock.exists():
        pore = np.load(source_pore).astype(bool)
        rock = np.load(source_rock).astype(bool)
        source = str(stage2_dir.resolve())
    else:
        temporary = output_dir / "_geometry_generation"
        static_cfg = {
            "seed": config["seed"],
            "volume": {
                **config["volume"],
                "pore_attenuation_mm_inv": config["volume"]["air_attenuation_mm_inv"],
                "porespy_generator": "porespy.generators.blobs",
            },
        }
        generate_porous_phantom(static_cfg, temporary)
        pore = np.load(temporary / "ground_truth" / "pore_mask.npy").astype(bool)
        rock = ~pore
        source = "deterministically regenerated with Stage-2 parameters"
        shutil.rmtree(temporary)
    if pore.shape != expected or not np.array_equal(rock, ~pore):
        raise ValueError("Stage-2 pore/rock geometry is incompatible with Stage 3")
    return pore, rock, source


def generate_dynamic_phantom(config: dict, output_dir: Path, stage2_dir: Path) -> dict:
    try:
        import porespy as ps
    except ImportError as exc:
        raise RuntimeError("PoreSpy is required for Stage 3") from exc

    np.random.seed(int(config["seed"]))
    pore, rock, geometry_source = _obtain_stage2_geometry(config, output_dir, stage2_dir)
    inlet = np.zeros_like(pore, dtype=bool)
    inlet[0] = pore[0]
    if not inlet.any():
        raise ValueError("The selected z-min inlet contains no pore voxels")
    result = ps.filters.ibip(im=pore, inlets=inlet, maxiter=int(config["invasion"]["maxiter"]))
    sequence = np.asarray(result.inv_sequence, dtype=np.int32)
    reachable = pore & (sequence > 0)
    ranks = np.zeros_like(sequence, dtype=np.float32)
    unique_steps = np.unique(sequence[reachable])
    if unique_steps.size == 0:
        raise RuntimeError("IBIP did not invade any pore voxels")
    # Map invasion events to (0,1]; t=0 is rigorously empty.
    positions = np.searchsorted(unique_steps, sequence[reachable]) + 1
    ranks[reachable] = positions.astype(np.float32) / float(unique_steps.size)

    state_times = np.linspace(0.0, 1.0, int(config["invasion"]["stored_states"]), dtype=np.float32)
    fluid_masks = np.asarray([(ranks > 0) & (ranks <= t) for t in state_times], dtype=bool)
    if np.any(fluid_masks & ~pore) or np.any(np.diff(fluid_masks.astype(np.int8), axis=0) < 0):
        raise RuntimeError("Invalid non-pore or non-monotonic fluid occupancy")
    saturation = fluid_masks.reshape(len(state_times), -1).sum(axis=1) / float(pore.sum())
    # Labels: 0=air/empty pore, 1=fluid, 2=rock.  Keeping a label volume for
    # every stored state makes material occupancy independently auditable.
    material_labels = np.zeros(fluid_masks.shape, dtype=np.uint8)
    material_labels[:, rock] = 2
    material_labels[fluid_masks] = 1

    root = output_dir / "ground_truth"
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "pore_mask.npy", pore)
    np.save(root / "rock_mask.npy", rock)
    np.save(root / "inlet_mask.npy", inlet)
    np.save(root / "invasion_sequence.npy", sequence)
    np.save(root / "invasion_time_normalized.npy", ranks)
    np.save(root / "state_times_normalized.npy", state_times)
    np.save(root / "fluid_masks.npy", fluid_masks)
    np.save(root / "material_labels.npy", material_labels)
    np.save(root / "saturation.npy", saturation.astype(np.float32))
    representative = root / "representative_volumes"
    representative.mkdir(parents=True, exist_ok=True)
    for index in sorted({0, len(state_times) // 4, len(state_times) // 2, 3 * len(state_times) // 4, len(state_times) - 1}):
        labels = material_labels[index]
        volume = np.zeros(rock.shape, dtype=np.float32)
        volume[labels == 1] = np.float32(config["volume"]["fluid_attenuation_mm_inv"])
        volume[labels == 2] = np.float32(config["volume"]["rock_attenuation_mm_inv"])
        np.save(representative / f"mu_state_{index:03d}.npy", volume)
    metadata = {
        "array_order": "zyx",
        "geometry_source": geometry_source,
        "seed": int(config["seed"]),
        "algorithm": config["invasion"]["algorithm"],
        "inlet": config["invasion"]["inlet"],
        "pore_voxels": int(pore.sum()),
        "rock_voxels": int(rock.sum()),
        "reachable_pore_voxels": int(reachable.sum()),
        "unreachable_pore_voxels": int((pore & ~reachable).sum()),
        "initial_saturation": float(saturation[0]),
        "final_saturation": float(saturation[-1]),
        "coefficients_mm_inv": {
            "rock": config["volume"]["rock_attenuation_mm_inv"],
            "air": config["volume"]["air_attenuation_mm_inv"],
            "fluid": config["volume"]["fluid_attenuation_mm_inv"],
        },
        "material_label_encoding": {"0": "air/empty pore", "1": "fluid", "2": "rock"},
        "state_transition": config["invasion"]["state_transition"],
        "physical_speed_claimed": False,
    }
    write_json(root / "metadata.json", metadata)
    write_json(root / "completed.json", {"metadata": str(root / "metadata.json")})
    return metadata


def state_index_for_time(state_times: np.ndarray, time_normalized: float) -> int:
    """Right-continuous lookup: use the latest stored state at or before t."""
    return int(np.clip(np.searchsorted(state_times, time_normalized, side="right") - 1, 0, len(state_times) - 1))


def attenuation_at_time(output_dir: Path, config: dict, time_normalized: float) -> tuple[np.ndarray, np.ndarray, int]:
    root = output_dir / "ground_truth"
    times = np.load(root / "state_times_normalized.npy")
    index = state_index_for_time(times, float(time_normalized))
    fluid = np.load(root / "fluid_masks.npy", mmap_mode="r")[index].astype(bool)
    rock = np.load(root / "rock_mask.npy").astype(bool)
    volume = np.full(rock.shape, config["volume"]["air_attenuation_mm_inv"], dtype=np.float32)
    volume[fluid] = np.float32(config["volume"]["fluid_attenuation_mm_inv"])
    volume[rock] = np.float32(config["volume"]["rock_attenuation_mm_inv"])
    return volume, fluid, index
