from abstract_layer import Layer
from grf_init import GRF_Init
import numpy as np

class Input_Layer(Layer):
    
    def __init__(self, input_size : int, output_size : int, mod : float) -> None:
        super().__init__(input_size, output_size)
        self.mod : float = mod

    def init_weights(self, window : np.ndarray, TS : float) -> None:
        grf_init = GRF_Init(window, self.shape[0], self.shape[1], TS, self.mod)
        self.weights = grf_init.init_weights()
