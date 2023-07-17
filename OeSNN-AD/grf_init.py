import numpy as np


class GRF_Init:

    def __init__(self, window: np.ndarray, input_size: int, TS: float, mod: int) -> None:
        self.min_w_i: float = window.min()
        self.max_w_i: float = window.max()
        self.xt: float = window[-1]

        self.input_size = input_size
        self.TS = TS
        self.mod = mod

    def _get_width_vec(self) -> np.ndarray:
        return np.repeat((self.max_w_i - self.min_w_i) / (self.input_size - 2), self.input_size)

    def _get_center_vec(self) -> np.ndarray:
        return np.array([self.min_w_i + (2*j - 3) / 2 * (self.max_w_i - self.min_w_i) / (self.input_size - 2)
                         for j in range(self.input_size)])

    def _get_excitation(self, width_v: np.ndarray, center_v: np.ndarray):
        rep_xt = np.repeat(self.xt, self.input_size)
        return np.exp(-0.5 * ((rep_xt - center_v) / width_v) ** 2)

    def _get_firing_time(self, excitation: np.ndarray) -> np.ndarray:
        return self.TS * (np.ones(self.input_size) - excitation)

    def _get_order(self, firings_times: np.ndarray) -> np.ndarray:
        arg_sorted = np.argsort(firings_times)
        return np.array([int(np.argwhere(arg_sorted == i))
                         for i in range(self.input_size)])

    def get_order(self) -> np.ndarray:

        width_v = self._get_width_vec()
        center_v = self._get_center_vec()
        excitation = self._get_excitation(width_v, center_v)
        firings_times = self._get_firing_time(excitation)

        return self._get_order(firings_times)
