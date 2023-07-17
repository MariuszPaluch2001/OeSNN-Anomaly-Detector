from neuron import Output_Neuron

import numpy as np

from pytest import approx


def test_update_neuron():
    updated_neuron = Output_Neuron(
        np.array([0.25, 0.25, 0.25]), 0.25, 0.1, 1, 0.25, 0.75, 2)
    candidate_neuron = Output_Neuron(
        np.array([0.75, 0.75, 0.75]), 0.5, 0.3, 1, 0.75, 1.0, 2)

    updated_neuron.update_neuron(candidate_neuron)

    assert updated_neuron.M == 2
    assert updated_neuron.output_value == approx(0.2)
    assert updated_neuron.addition_time == approx(0.5)
    np.testing.assert_array_almost_equal(
        updated_neuron.weights, np.array([0.5, 0.5, 0.5]), decimal=3)
