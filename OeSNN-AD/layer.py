import numpy as np
from typing import List, Tuple

from neuron import Neuron
from grf_init import GRF_Init
from neuron import Input_Neuron, Output_Neuron


class Layer:

    def __init__(self, neurons_n: int) -> None:
        self.neurons_n = neurons_n

        self.neurons: List[Neuron]


class Input_Layer(Layer):

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size)

        self.neurons: List[Input_Neuron] = [
            Input_Neuron(0.0) for _ in range(input_size)]
        self.orders: np.ndarray = None

    def set_orders(self, window: np.ndarray, TS: float, mod: float):
        grf = GRF_Init(window, self.neurons_n, TS, mod)

        self.orders = grf.get_order()


class Output_Layer(Layer):

    def __init__(self, neurons_n: int, max_output_size: int) -> None:
        super().__init__(neurons_n)

        self.max_outpt_size = max_output_size
        self.neurons: List[Output_Neuron] = []

    def make_candidate(self, window: np.ndarray, order: np.ndarray, mod: float,
                       C: float, neuron_age: int):

        weights = np.array([mod ** o for o in order])
        output_value = np.random.normal(np.mean(window), np.std(window))
        PSP_max = (weights * (mod ** order)).sum()
        gamma = C * PSP_max

        return Output_Neuron(weights, gamma,
                             output_value, 1, neuron_age,
                             0, PSP_max)

    def find_most_similar(self, candidate_neuron: Output_Neuron) -> Tuple[Output_Neuron | None, float]:
        if not self.neurons:
            return None, np.Inf

        def dist_f(n: Output_Neuron): return np.linalg.norm(
            n.weights - candidate_neuron.weights)
        most_similar_neuron = min(self.neurons, key=dist_f)
        min_distance = dist_f(most_similar_neuron)

        return most_similar_neuron, min_distance
