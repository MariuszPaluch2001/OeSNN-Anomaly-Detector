from grf_init import GRF_Init
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
    grf = GRF_Init(WINDOW, 7, 10, 1, 2)
    
    result = grf._get_width_vec()
    correct = np.repeat(0.18, 7)
    np.testing.assert_array_equal(result, correct)