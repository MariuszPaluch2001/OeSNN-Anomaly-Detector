"""
    Module docstring
"""

import numpy as np


class GRFInit:
    """
        Class docstring
    """

    def __init__(self, window: np.ndarray, input_size: int, ts_factor: float,
                 mod: int, beta: float) -> None:
        self.min_w_i: float = window.min()
        self.max_w_i: float = window.max()
        self.window_head: float = window[-1]

        self.input_size = input_size
        self.ts_factor = ts_factor
        self.mod = mod
        self.beta = beta

    def _get_width_vec(self) -> np.ndarray:
        """
            Method docstring
        """
        return np.repeat((self.max_w_i - self.min_w_i) / ((self.input_size - 2) * self.beta),
                         self.input_size)

    def _get_center_vec(self) -> np.ndarray:
        """
            Method docstring
        """
        return np.array([(self.min_w_i + ((2*j - 3) / 2) *
                        (self.max_w_i - self.min_w_i) / (self.input_size - 2))
            for j in range(self.input_size)])

    def _get_excitation(self, width_v: np.ndarray, center_v: np.ndarray):
        """
            Method docstring
        """
        rep_xt = np.repeat(self.window_head, self.input_size)
        return np.exp(-0.5 * ((rep_xt - center_v) / width_v) ** 2)

    def _get_firing_time(self, excitation: np.ndarray) -> np.ndarray:
        """
            Method docstring
        """
        return self.ts_factor * (np.ones(self.input_size) - excitation)

    def _get_order(self, firings_times: np.ndarray) -> np.ndarray:
        """
            Method docstring
        """
        arg_sorted = np.argsort(firings_times)
        return np.array([int(np.argwhere(arg_sorted == i))
                         for i in range(self.input_size)])

    def get_order(self) -> np.ndarray:
        """
            Method docstring
        """
        width_v = self._get_width_vec()
        center_v = self._get_center_vec()
        excitation = self._get_excitation(width_v, center_v)
        firings_times = self._get_firing_time(excitation)

        return self._get_order(firings_times)
