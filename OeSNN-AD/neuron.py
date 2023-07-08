import numpy as np

class Neuron:
    def __init__(self, id : int) -> None:
        self.id = id

class Input_Neuron(Neuron):
    def __init__(self, id : int, firing_time : float) -> None:
        super().__init__(id)
        self.firing_time = firing_time

class Output_Neuron(Neuron):
    def __init__(self, id : int, weights : np.ndarray, gamma : float, 
                 output_value : float, M: float, addition_time : float, 
                 PSP : float, max_PSP : float = 0.0) -> None:
        super().__init__(id)
        self.weights = weights
        self.gamma = gamma
        self.output_value = output_value
        self.M = M
        self.addition_time = addition_time
        self.PSP = PSP
        self.max_PSP = max_PSP
    
    def update_neuron(self, candidate_neuron : 'Output_Neuron'):
        self.weights = (candidate_neuron.weights + self.M * self.weights) / (self.M + 1)
        self.output_value = (candidate_neuron.output_value + self.M * self.output_value) / (self.M + 1)
        self.addition_time = (candidate_neuron.addition_time + self.M * self.addition_time) / (self.M + 1)
        self.m += 1