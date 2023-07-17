from output_layer import Output_Layer
from neuron import Output_Neuron

import numpy as np

from pytest import approx

WINDOW = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test_make_candidate():
    output_layer = Output_Layer(0, 10)

    order, mod, C, neuron_age = np.array([6, 5, 3, 1, 0, 2, 4]), 0.5, 0.5, 10

    candidate = output_layer.make_candidate(WINDOW, order, mod, C, neuron_age)

    correct_weights = np.array(
        [0.015625, 0.03125, 0.125, 0.5, 1, 0.25, 0.0625])

    assert type(candidate.output_value) is float

    assert candidate.addition_time == neuron_age
    assert candidate.PSP == 0
    assert candidate.M == 1

    np.testing.assert_array_almost_equal(
        candidate.weights, correct_weights, decimal=3)
    assert candidate.max_PSP == approx(1.333, abs=1e-3)
    assert candidate.gamma == approx(0.666, abs=1e-3)


def test_find_most_similar_without_neurons():
    output_layer = Output_Layer(0, 10)
    neuron = Output_Neuron(
        np.array([0.25, 0.25, 0.25]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    result_n, distance = output_layer.find_most_similar(neuron)

    assert result_n is None
    assert np.isinf(distance)


def test_find_most_similar_with_neurons():
    output_layer = Output_Layer(0, 10)

    neuron1 = Output_Neuron(
        np.array([0.26, 0.26, 0.26]), 0.25, 0.1, 1, 0.25, 0.75, 2)
    neuron2 = Output_Neuron(
        np.array([1.0, 1.0, 1.0]), 0.25, 0.1, 1, 0.25, 0.75, 2)
    neuron3 = Output_Neuron(
        np.array([0.0, 0.0, 0.0]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    output_layer.neurons.extend([neuron1, neuron2, neuron3])

    c_neuron = Output_Neuron(
        np.array([0.25, 0.25, 0.25]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    result_n, distance = output_layer.find_most_similar(c_neuron)

    assert result_n is neuron1
    assert distance == approx(0.0173, abs=1e-4)
