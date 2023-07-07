from input_layer import Input_Layer
import numpy as np

WINDOW = np.array([0.5, 
                   0.3, 
                   0.4, 
                   0.3, 
                   0.6, 
                   0.2, 
                   1.0, 
                   0.4, 
                   0.3, 
                   0.4, 
                   0.2, 
                   0.4, 
                   0.1, 
                   0.5])

def test_GRF_width_vector_calc():
    layer = Input_Layer(7, 10, 2)
    

    result = layer._GRF_width_vector_calc(WINDOW.min(), WINDOW.max())
    correct = np.repeat(0.18, 7)
    np.testing.assert_array_equal(result, correct)