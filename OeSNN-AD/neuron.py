"""
    Moduł zawiera definicje i implementacje neuronów.
"""
import numpy as np
import numpy.typing as npt


class Neuron:
    """
        Klasa bazowa neuronu.
    """

    def __init__(self) -> None:
        pass


class InputNeuron(Neuron):
    """
        Klasa wejściowego neuronu, dziedzicząca po bazowej klasie neuronu.
    """

    def __init__(self, firing_time: float, neuron_id: int = 0, order: int = 0) -> None:
        super().__init__()
        self.neuron_id = neuron_id
        self.firing_time = firing_time
        self.order = order

    def set_order(self, new_order: int):
        """
            Setter dla atrybtu order.
        """
        self.order = new_order


class OutputNeuron(Neuron):
    """
        Klasa wyjściowego neuronu, dziedzicząca po bazowej klasie neuronu.
    """

    def __init__(self, weights: npt.NDArray[np.float64], gamma: float,
                 output_value: float, modification_count: float, addition_time: float,
                 PSP: float, max_PSP: float) -> None:
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.output_value = output_value
        self.modification_count = modification_count
        self.addition_time = addition_time
        self.psp = PSP
        self.max_psp = max_PSP

    def __getitem__(self, index: int) -> float:
        return self.weights[index]

    def update_neuron(self, candidate_neuron: 'OutputNeuron') -> None:
        """
            Metoda służąca do aktualizacji neuronu na podstawie innego neuronu
            przekazywanego jako parametr.
        """
        self.weights = (candidate_neuron.weights +
                        self.modification_count * self.weights) / (self.modification_count + 1)
        self.output_value = ((candidate_neuron.output_value +
                             self.modification_count * self.output_value) /
                             (self.modification_count + 1))
        self.addition_time = ((
            candidate_neuron.addition_time + self.modification_count * self.addition_time) /
            (self.modification_count + 1))
        self.modification_count += 1

    def error_correction(self, window_head: float, ksi: float) -> None:
        """
            Dodaj docstring
        """
        # TODO: Dodać testy do tego
        self.output_value += (window_head - self.output_value) * ksi

    def error_calc(self, window_head: float) -> float:
        """
            Dodaj docstring
        """
        # TODO: dodaj testy do tego
        return np.abs(window_head - self.output_value)
