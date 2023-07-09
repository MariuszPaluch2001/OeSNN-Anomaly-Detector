from output_layer import Output_Layer

import numpy as np

from pytest import approx

WINDOW = np.array([0.5, 0.3, 0.4, 
                   0.3, 0.6, 0.2, 
                   1.0, 0.4, 0.3, 
                   0.4, 0.2, 0.4, 
                   0.1, 0.5])

def test_make_candidate():
    output_layer = Output_Layer(0, 10)
    
    order, mod, C, neuron_age = np.array([6,5,3,1,0,2,4]), 0.5, 0.5, 10

    candidate = output_layer.make_candidate(WINDOW, order, mod, C, neuron_age)
    
    correct_weights = np.array([0.015625, 0.03125, 0.125, 0.5, 1, 0.25, 0.0625])

    assert type(candidate.output_value) is float

    assert candidate.addition_time == neuron_age
    assert candidate.PSP == 0
    assert candidate.M == 1
    
    np.testing.assert_array_almost_equal(candidate.weights, correct_weights, decimal = 3)
    assert candidate.max_PSP == approx(1.333, abs=1e-3)
    assert candidate.gamma == approx(0.666, abs=1e-3)