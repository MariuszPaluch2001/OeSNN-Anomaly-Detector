from abstract_layer import Layer
import numpy as np

class Input_Layer(Layer):
    
    def __init__(self, input_size : int, output_size : int, mod : float) -> None:
        super().__init__(input_size, output_size)
        self.mod : float = mod
    
    def _GRF_width_vector_calc(self, min_w_i : float, max_w_i : float) -> np.ndarray:
        return np.repeat((max_w_i - min_w_i) / (self.shape[0] - 2), self.shape[0])

    def _GRF_center_vector_calc(self, min_w_i : float, max_w_i : float) -> np.ndarray:
        return np.array([min_w_i + (2*j - 3) / 2 * (max_w_i - min_w_i) / (self.shape[0] - 2) 
                         for j in range(self.shape[0])])

    def _GRF_excitation_calc(self, x_t : float, grf_width : np.ndarray, grf_center : np.ndarray):
        rep_xt = np.repeat(x_t, self.shape[0])
        return np.exp(-0.5 * ((rep_xt - grf_center) / grf_width) ** 2)
    
    def _GRF_firing_time_calc(self, excitation : np.ndarray, TS : float) -> np.ndarray:
        return TS * (np.ones(self.shape[0]) - excitation)
    
    def _GRF_order_calc(self, firings_times : np.ndarray) -> np.ndarray:
        return np.array([int(np.argwhere(np.argsort(firings_times)) == i) 
                         for i in range(self.shape[0])])
    
    def _GRF_initialize_weights(self, orders : np.ndarray) -> None:
        for i in range(self.shape[0]):
            self.weights[i] = np.repeat(self.mod ** orders[i], self.shape[1])

    def _GRF_initialization(self, window : np.ndarray, TS : float) -> None:
        x_t = window[-1]

        min_w_i = window.min()
        max_w_i = window.max()

        grf_width_vec = self._GRF_width_vector_calc(min_w_i, max_w_i)
        grf_center_vec = self._GRF_center_vector_calc(min_w_i, max_w_i)
        excitation = self._GRF_excitation_calc(x_t, grf_width_vec, grf_center_vec)
        firings_times = self._GRF_firing_time_calc(excitation, TS)
        orders = self._GRF_order_calc(firings_times)
        
        self._GRF_initialize_weights(orders)
