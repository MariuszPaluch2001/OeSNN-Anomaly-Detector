from layer import Output_Layer
from neuron import Output_Neuron

import numpy as np

from pytest import approx

WINDOW = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test_make_candidate():
    output_layer = Output_Layer(10)

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
    output_layer = Output_Layer(10)
    candidate = Output_Neuron(
        np.array([0.25, 0.25, 0.25]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    neuron_result, distance = output_layer.find_most_similar(candidate)

    assert neuron_result is None
    assert np.isinf(distance)


def test_find_most_similar_with_neurons():
    output_layer = Output_Layer(10)

    neuron_out1 = Output_Neuron(
        np.array([0.26, 0.26, 0.26]), 0.25, 0.1, 1, 0.25, 0.75, 2)
    neuron_out2 = Output_Neuron(
        np.array([1.0, 1.0, 1.0]), 0.25, 0.1, 1, 0.25, 0.75, 2)
    neuron_out3 = Output_Neuron(
        np.array([0.0, 0.0, 0.0]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    output_layer.add_new_neuron(neuron_out1)
    output_layer.add_new_neuron(neuron_out2)
    output_layer.add_new_neuron(neuron_out3)

    c_neuron = Output_Neuron(
        np.array([0.25, 0.25, 0.25]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    neuron_result, distance = output_layer.find_most_similar(c_neuron)

    assert neuron_result == neuron_out1
    assert distance == approx(0.0173, abs=1e-4)


def test_reset_psp():
    output_layer = Output_Layer(10)
    neuron_out1 = Output_Neuron(None, None, None, None, None, 5, 10)
    neuron_out2 = Output_Neuron(None, None, None, None, None, 6, 10)
    neuron_out3 = Output_Neuron(None, None, None, None, None, 7, 10)

    output_layer.add_new_neuron(neuron_out1)
    output_layer.add_new_neuron(neuron_out2)
    output_layer.add_new_neuron(neuron_out3)

    output_layer.reset_psp()

    for neuron in output_layer.neurons:
        assert neuron.PSP == 0.0


def test_add_new_neuron():
    output_layer = Output_Layer(10)
    assert output_layer.num_neurons == 0
    assert len(output_layer.neurons) == 0

    neuron_out1 = Output_Neuron(None, None, None, None, None, None, None)
    output_layer.add_new_neuron(neuron_out1)
    assert output_layer.num_neurons == 1
    assert len(output_layer.neurons) == 1
    assert output_layer.neurons[0] == neuron_out1

    neuron_out2 = Output_Neuron(None, None, None, None, None, None, None)
    output_layer.add_new_neuron(neuron_out2)
    assert output_layer.num_neurons == 2
    assert len(output_layer.neurons) == 2
    assert output_layer.neurons[1] == neuron_out2


def test_replace_oldest():
    output_layer = Output_Layer(10)

    neuron_out1 = Output_Neuron(None, None, None, None, 1, None, None)
    neuron_out2 = Output_Neuron(None, None, None, None, 2, None, None)
    neuron_out3 = Output_Neuron(None, None, None, None, 3, None, None)
    candidate = output_layer.make_candidate(
        WINDOW, np.array([1, 2, 3]), 0.0, 0.0, 10)

    output_layer.add_new_neuron(neuron_out1)
    output_layer.add_new_neuron(neuron_out2)
    output_layer.add_new_neuron(neuron_out3)

    output_layer.replace_oldest(candidate)

    assert neuron_out1 not in output_layer.neurons
    assert candidate in output_layer.neurons
