from input_layer import Input_Layer
from output_layer import Output_Layer
import numpy as np

class OeSNN_AD:

    def __init__(self, stream: np.ndarray, window_size : int, 
                 input_neurons_n : int, output_neurons_n : int, 
                 TS : float, mod : float, C : float) -> None:
        
        self.stream : np.ndarray = stream
        self.stream_len : int = self.stream.shape[0]
        self.window_size : int = window_size

        self.input_layer : Input_Layer = Input_Layer(input_neurons_n, output_neurons_n)
        self.output_layer : Output_Layer = Output_Layer(output_neurons_n)
    
        self.TS = TS
        self.mod = mod
        self.C = C
        
        self.gamma = self.C * (1 - self.mod**(2*input_neurons_n)) / (1 - self.mod**2)

    def predict(self) -> np.ndarray:
        current_no_size = 0
        window = self.stream[0:self.window_size]
        pred_values = np.random.normal(np.mean(window), np.std(window), self.window_size).tolist()
        anomaly_result = [False for _ in range(self.window_size)]

        for t in range(self.window_size + 1, self.stream_len):
            window = self.stream[t - self.window_size : t]
            is_anomaly = self._anomaly_classification(window)
            anomaly_result.append(is_anomaly)

            self._learning(window)
        
        return np.array(anomaly_result)
    
    def _anomaly_classification(self, window : np.ndarray) -> bool:
        self.input_layer.init_weights(window)
    
    def _learning(self, window : np.ndarray) -> None:
        ...

    def _error_correction(self) -> None:
        ...