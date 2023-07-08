from abstract_layer import Layer
from neuron import Output_Neuron

from typing import List

class Output_Layer(Layer):
    
    def __init__(self, neurons_n : int, max_output_size : int) -> None:
        super().__init__(neurons_n)
        
        self.max_outpt_size = max_output_size
        self.neurons : List[Output_Neuron] = []
