import numpy as np
from typing import Tuple

class Layer:
    
    def __init__(self, input_size : int, output_size : int) -> None:
        self.shape : Tuple[int, int] = (input_size, output_size)
        self.weights : np.ndarray = np.zeros((input_size, output_size))