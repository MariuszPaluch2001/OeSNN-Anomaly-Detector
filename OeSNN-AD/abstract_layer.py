import numpy as np
from typing import Tuple, List

from neuron import Neuron


class Layer:

    def __init__(self, neurons_n: int) -> None:
        self.neurons_n = neurons_n

        self.neurons: List[Neuron]
