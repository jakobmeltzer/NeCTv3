import numpy as np

from experiments.attenuation_validation.dynamic_config import dynamic_experiment_config
from experiments.attenuation_validation.dynamic_phantom import state_index_for_time
from experiments.attenuation_validation.dynamic_projections import acquisition_arrays


def test_acquisition_alignment_and_angle_mapping():
    cfg = dynamic_experiment_config(num_projections=12, num_revolutions=3, epochs=1, stored_states=5)
    times, tigre, nect = acquisition_arrays(cfg)
    assert len(times) == len(tigre) == len(nect) == 12
    assert np.all(np.diff(times) > 0)
    assert np.max(np.abs(np.angle(np.exp(1j * (nect - tigre - np.pi))))) < 2e-6


def test_discrete_state_lookup_is_right_continuous():
    states = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert state_index_for_time(states, 0.0) == 0
    assert state_index_for_time(states, 0.49) == 1
    assert state_index_for_time(states, 0.5) == 2
    assert state_index_for_time(states, 1.0) == 4
