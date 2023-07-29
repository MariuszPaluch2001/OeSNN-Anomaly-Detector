"""
    Moduł testujący metody klas neuronów.
"""

from neuron import OutputNeuron

import numpy as np

from pytest import approx


def test_update_neuron():
    """
        Test sprawdzający czy neuron faktycznie jest aktualizowany.
    """
    updated_neuron = OutputNeuron(
        np.array([0.25]*3), 0.25, 0.1, 1, 0.25, 0.75, 2)
    candidate_neuron = OutputNeuron(
        np.array([0.75]*3), 0.5, 0.3, 1, 0.75, 1.0, 2)

    updated_neuron.update_neuron(candidate_neuron)

    assert updated_neuron.modification_count == 2
    assert updated_neuron.output_value == approx(0.2)
    assert updated_neuron.addition_time == approx(0.5)
    np.testing.assert_array_almost_equal(
        updated_neuron.weights, np.array([0.5]*3), decimal=3)


def test__get_item__():
    """
        Test sprawdzający czy indeksowanie neuronu faktycznie zwraca wartości
        w wektorze wag.
    """
    neuron = OutputNeuron(
        np.array([0.25, 0.5, 0.75]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    assert neuron[0] == 0.25
    assert neuron[1] == 0.5
    assert neuron[2] == 0.75
