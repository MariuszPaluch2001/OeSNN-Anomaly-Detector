from abstract_layer import Layer
from grf_init import GRF_Init
from neuron import Input_Neuron

import numpy as np
from typing import List

class Input_Layer(Layer):
    
    def __init__(self, input_size : int) -> None:
        super().__init__(input_size)
        
        self.neurons : List[Input_Neuron] = [Input_Neuron(0.0) for _ in range(input_size)]
        self.orders : np.ndarray = None

    def set_orders(self, window : np.ndarray, TS : float, mod : float):
        grf = GRF_Init(window, self.neurons_n, TS, mod)
        
        self.orders = grf.get_order()
