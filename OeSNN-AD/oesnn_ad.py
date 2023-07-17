from input_layer import Input_Layer
from output_layer import Output_Layer
from neuron import Output_Neuron

import numpy as np

class OeSNN_AD:

    def __init__(self, stream: np.ndarray, window_size : int, 
                 input_neurons_n : int, output_neurons_n : int, 
                 TS : float, mod : float, C : float, epsilon : float) -> None:
        
        self.stream = stream
        self.stream_len = self.stream.shape[0]
        self.window_size = window_size

        self.input_layer : Input_Layer = Input_Layer(input_neurons_n)
        self.output_layer : Output_Layer = Output_Layer(0, output_neurons_n)
    
        self.TS = TS
        self.mod = mod
        self.C = C
        
        self.gamma = self.C * (1 - self.mod**(2*input_neurons_n)) / (1 - self.mod**2)
        self.epsilon  = epsilon

    def predict(self) -> np.ndarray:
        current_no_size = 0
        window = self.stream[0:self.window_size]
        pred_values = np.random.normal(np.mean(window), np.std(window), self.window_size).tolist()
        anomaly_result = [False for _ in range(self.window_size)]

        for t in range(self.window_size + 1, self.stream_len):
            self.input_layer.set_orders(window, self.TS, self.mod)
            
            window = self.stream[t - self.window_size : t]
            is_anomaly = self._anomaly_classification(window)
            anomaly_result.append(is_anomaly)

            self._learning(window, t)
        
        return np.array(anomaly_result)
    
    def _anomaly_classification(self, error_values : np.ndarray, 
                                anom_classification: np.ndarray) -> bool:
        err_t = error_values[-1]
        
        err_anom = [err for err, classification 
                        in zip(error_values[-(self.window_size - 1):-1], anom_classification[-(self.window_size - 1):]) if not classification]

        return not (
            (not err_anom) or (err_t - np.mean(err_anom) < np.std(err_anom) * self.epsilon)
        )
    
    def _learning(self, window : np.ndarray, neuron_age : int) -> None:
        candidate = self.output_layer.make_candidate(window, self.input_layer.orders, 
                                                     self.mod, self.C, neuron_age)

    def _error_correction(self) -> None:
        ...

    def _fires_first(self) -> Output_Neuron | None:
        to_fire = []
        
        for n in self.output_layer.neurons:
            n.PSP = 0
        
        for idx_in in range(len(self.input_layer.neurons)):
            for n_out in self.output_layer.neurons:
                n_out.PSP += n_out.weights[idx_in] * (self.mod ** self.input_layer.orders[idx_in])

                if n_out.PSP > self.gamma:
                    to_fire.append(n_out)
                
            if to_fire:
                return max(to_fire, key = lambda x: x.PSP)