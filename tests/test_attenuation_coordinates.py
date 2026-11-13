import numpy as np

from experiments.attenuation_validation.coordinates import (
    nect_angles_for_tigre_world,
    nect_model_volume_to_tigre_world,
    tigre_world_to_nect_model_points,
)


def test_rz_pi_point_mapping_is_self_inverse():
    points = np.array([[0.1, 0.2, 0.3], [0.8, 0.6, 0.4]])
    assert np.allclose(
        tigre_world_to_nect_model_points(tigre_world_to_nect_model_points(points)),
        points,
    )


def test_model_volume_mapping_reverses_y_and_x_only():
    volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    mapped = nect_model_volume_to_tigre_world(volume)
    assert np.array_equal(mapped, volume[:, ::-1, ::-1])
    assert mapped.flags.c_contiguous


def test_new_training_angle_mapping_is_pi_offset():
    angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    expected = np.array([np.pi, 3 * np.pi / 2, 0.0, np.pi / 2])
    assert np.allclose(nect_angles_for_tigre_world(angles), expected)
