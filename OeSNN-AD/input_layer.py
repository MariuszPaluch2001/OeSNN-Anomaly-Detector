from abstract_layer import Layer
from grf_init import GRF_Init
from neuron import Input_Neuron

import numpy as np
from typing import List

class Input_Layer(Layer):
    
    def __init__(self, input_size : int, output_size : int) -> None:
        super().__init__(input_size, output_size)
        
        self.neurons : List[Input_Neuron] = [Input_Neuron(id, 0.0) for id in range(input_size)]
