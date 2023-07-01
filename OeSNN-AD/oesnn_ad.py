from input_layer import Input_Layer
from output_layer import Output_Layer
import numpy as np

class OeSNN_AD:

    def __init__(self, stream: np.ndarray, input_neurons_n : int, output_neurons_n : int, window_size : int) -> None:
        self.stream : np.ndarray = stream
        self.window_size : int = window_size

        self.input_layer : Input_Layer = Input_Layer(input_neurons_n, output_neurons_n)
        self.output_layer : Output_Layer = Output_Layer(output_neurons_n)
    
    def predict() -> np.ndarray:
        ...

    def _anomaly_classification(predict_value : float, actual_value : float) -> bool:
        ...
    
    def _error_correction() -> None:
        ...