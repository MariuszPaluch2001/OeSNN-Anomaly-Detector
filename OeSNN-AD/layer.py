"""
    Module docstring
"""

from typing import List, Tuple, Generator
import numpy as np

from neuron import Neuron
from neuron import InputNeuron, OutputNeuron
from grf_init import GRFInit


class Layer:
    """
        Class docstring
    """

    def __init__(self, num_neurons: int) -> None:
        self.num_neurons = num_neurons

        self.neurons: List[Neuron]

    def __iter__(self) -> Generator[Neuron, None, None]:
        """
            Method docstring
        """
        for neuron in self.neurons:
            yield neuron

    def __len__(self) -> int:
        """
            Method docstring
        """
        return len(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        """
            Method docstring
        """
        return self.neurons[index]


class InputLayer(Layer):
    """
        Class docstring
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size)

        self.neurons: List[InputNeuron] = [
            InputNeuron(0.0, id) for id in range(input_size)]

    def __iter__(self) -> Generator[InputNeuron, None, None]:
        """
            Method docstring
        """
        return super().__iter__()

    def __getitem__(self, index: int) -> InputNeuron:
        """
            Method docstring
        """
        return super().__getitem__(index)

    @property
    def orders(self):
        """
            Method docstring
        """
        return np.array([neuron.order for neuron in self.neurons])

    def set_orders(self, window: np.ndarray, ts_coef: float, mod: float, beta: float) -> None:
        """
            Method docstring
        """
        grf = GRFInit(window, self.num_neurons, ts_coef, mod, beta)

        for neuron, new_order in zip(self.neurons, grf.get_order()):
            neuron.set_order(new_order)


class OutputLayer(Layer):
    """
        Class docstring
    """

    def __init__(self, max_output_size: int) -> None:
        super().__init__(0)

        self.max_outpt_size = max_output_size
        self.neurons: List[OutputNeuron] = []

    def __iter__(self) -> Generator[OutputNeuron, None, None]:
        """
            Method docstring
        """
        return super().__iter__()

    def __getitem__(self, index: int) -> OutputNeuron:
        """
            Method docstring
        """
        return super().__getitem__(index)

    def make_candidate(self, window: np.ndarray, order: np.ndarray, mod: float,
                       c_coef: float, neuron_age: int) -> OutputNeuron:
        """
            Method docstring
        """
        weights = np.array([mod ** o for o in order])
        output_value = np.random.normal(np.mean(window), np.std(window))
        psp_max = (weights * (mod ** order)).sum()
        gamma = c_coef * psp_max

        return OutputNeuron(weights, gamma,
                             output_value, 1, neuron_age,
                             0, psp_max)

    def find_most_similar(self,
                          candidate_neuron: OutputNeuron) -> Tuple[OutputNeuron | None, float]:
        """
            Method docstring
        """
        if not self.neurons:
            return None, np.Inf

        def dist_f(neuron: OutputNeuron) -> float:
            return np.linalg.norm(neuron.weights - candidate_neuron.weights)
        most_similar_neuron = min(self.neurons, key=dist_f)
        min_distance = dist_f(most_similar_neuron)

        return most_similar_neuron, min_distance

    def add_new_neuron(self, neuron: OutputNeuron) -> None:
        """
            Method docstring
        """
        self.neurons.append(neuron)
        self.num_neurons += 1

    def replace_oldest(self, candidate: OutputNeuron) -> None:
        """
            Method docstring
        """
        oldest = min(self.neurons, key=lambda n: n.addition_time)
        self.neurons.remove(oldest)
        self.neurons.append(candidate)

    def reset_psp(self) -> None:
        """
            Method docstring
        """
        for neuron in self.neurons:
            neuron.psp = 0
