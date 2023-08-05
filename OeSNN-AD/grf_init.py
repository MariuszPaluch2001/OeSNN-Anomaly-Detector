"""
    Moduł zawierający implementację i definicję klasy GRFInit.
"""

import numpy as np


class GRFInit:
    """
        Klasa zawiera zbiór funkcji, których celem jest inicjalizacja
        kolejności wystrzeliwania neuronów w warstwie wejściowej.

        Obiekt klasy jest definiowany na nowo w czasie każdej iteracji algorytmu.
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
            Metoda do obliczania wektora szerokości GRF dla wszystkich neuronów w warstwie.
        """
        return np.repeat((self.max_w_i - self.min_w_i) / ((self.input_size - 2) * self.beta),
                         self.input_size)

    def _get_center_vec(self) -> np.ndarray:
        """
            Metoda od obliczania wektora centrów GRF dla wszystkich neuronów w warstwie.
        """
        return (self.min_w_i + ((2*np.arange(0, self.input_size, 1) - 3) / 2) *
                (self.max_w_i - self.min_w_i) / (self.input_size - 2))

    def _get_excitation(self, width_v: np.ndarray, center_v: np.ndarray):
        """
            Metoda do obliczania wektora ekscytacji GRF dla wszystkich neuronów w warstwie.
        """
        rep_xt = np.repeat(self.window_head, self.input_size)
        return np.exp(-0.5 * ((rep_xt - center_v) / width_v) ** 2)

    def _get_firing_time(self, excitation: np.ndarray) -> np.ndarray:
        """
            Metoda do obliczania wektora czasu wystrzeliwania GRF dla 
            wszystkich neuronów w warstwie.
        """
        return self.ts_factor * (np.ones(self.input_size) - excitation)

    def _get_order(self, firings_times: np.ndarray) -> np.ndarray:
        """
            Metoda do obliczania wektora kolejności wystrzeliwania GRF dla 
            wszystkich neuronów w warstwie.
        """
        arg_sorted = np.argsort(firings_times)
        orders = np.empty_like(arg_sorted)
        orders[arg_sorted] = np.arange(len(firings_times))
        return orders

    def get_order(self) -> np.ndarray:
        """
            Metoda jako publiczny interfejs do obliczania wektora porządku wystrzeliwania dla
            wszystkich neuronów w warstwie. 
        """
        width_v = self._get_width_vec()
        center_v = self._get_center_vec()
        excitation = self._get_excitation(width_v, center_v)
        firings_times = self._get_firing_time(excitation)

        return self._get_order(firings_times)
