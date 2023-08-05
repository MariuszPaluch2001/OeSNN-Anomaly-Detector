"""
    Moduł zawiera definicję i implementację klas warstw.
"""

from typing import List, Tuple, Generator
import numpy as np

from neuron import Neuron
from neuron import InputNeuron, OutputNeuron
from grf_init import GRFInit


class Layer:
    """
        Bazowa klasa tworząca interfejs dla klas dziedziczących.

        Oprócz atrybutów wspólnych dla wszystkich warst tworzy także,
        implementuje także metody magiczne umożliwiające iterowanie się,
        indeksowanie, a także liczenie ilości neuronów z pomocą wbudowanej
        funkcji len z poziomu obiektu, bez odwoływania się do atrybutu neurons.

        Nie powinna być tworzona jako osobny obiekt.
    """

    def __init__(self, num_neurons: int) -> None:
        self.num_neurons = num_neurons

        self.neurons: List[Neuron]

    def __iter__(self) -> Generator[Neuron, None, None]:
        for neuron in self.neurons:
            yield neuron

    def __len__(self) -> int:
        return len(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]


class InputLayer(Layer):
    """
        Klasa implementująca warstwę wejściową, dziedzicząca po bazowej
        klasie Layer.

        Klasa przechowuje i obsługuje listę neuronów wejściowych.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size)

        self.neurons: List[InputNeuron] = [
            InputNeuron(0.0, id) for id in range(input_size)]

    def __iter__(self) -> Generator[InputNeuron, None, None]:
        neurons = sorted(self.neurons, key=lambda neuron: neuron.order)
        for neuron in neurons:
            yield neuron

    def __getitem__(self, index: int) -> InputNeuron:
        return super().__getitem__(index)

    @property
    def orders(self):
        """
            Atrybut, który złącza kolejność wystrzylewania neuronów
            w jedną listę.
        """
        vectorized_get_order = np.vectorize(lambda neuron: neuron.order)
        return vectorized_get_order(self.neurons)

    def set_orders(self, window: np.ndarray, ts_coef: float, mod: float, beta: float) -> None:
        """
            Metoda służy do ustawienia dla każdego neuronu wejściowego
            nowej pozycji wystrzeliwania w warstwie.
        """
        grf = GRFInit(window, self.num_neurons, ts_coef, mod, beta)

        for neuron, new_order in zip(self.neurons, grf.get_order()):
            neuron.set_order(new_order)


class OutputLayer(Layer):
    """
        Klasa implementująca warstwę wyjściową, dziedzicząca po bazowej
        klasie Layer.

        Klasa przechowuje i obsługuje listę neuronów wyjściowych.
    """

    def __init__(self, max_output_size: int) -> None:
        super().__init__(0)

        self.max_outpt_size = max_output_size
        self.neurons: List[OutputNeuron] = []

    def __iter__(self) -> Generator[OutputNeuron, None, None]:
        return super().__iter__()

    def __getitem__(self, index: int) -> OutputNeuron:
        return super().__getitem__(index)

    def make_candidate(self, window: np.ndarray, order: np.ndarray, mod: float,
                       c_coef: float, neuron_age: int) -> OutputNeuron:
        """
            Metoda tworząca nowy neuron wyjściowy i ustawiająca jego składowe.
        """
        weights = mod ** order
        output_value = np.random.normal(np.mean(window), np.std(window))
        psp_max = (weights * (mod ** order)).sum()
        gamma = c_coef * psp_max

        return OutputNeuron(weights, gamma,
                            output_value, 1, neuron_age,
                            0, psp_max)

    def find_most_similar(self,
                          candidate_neuron: OutputNeuron) -> Tuple[OutputNeuron | None, float]:
        """
            Metoda zwracająca neuron mająca najmniejszą odległość euklidesową od
            neuronu kandydata, wraz z odległością. Gdy warstwa nie ma neuronów zwraca parę 
            None, np.inf.
        """
        if not self.neurons:
            return None, np.Inf

        def dist_f(neuron: OutputNeuron) -> float:
            return np.linalg.norm(neuron.weights - candidate_neuron.weights)
        most_similar_neuron = min(self.neurons, key=dist_f)
        min_distance = dist_f(most_similar_neuron)

        return most_similar_neuron, min_distance

    def add_new_neuron(self, neuron: OutputNeuron) -> None:
        """
            Metoda służy do dodawania nowego nowego neuronu, gdy
            liczba neuronów w warstwie jest poniżej maksymalnej.

            Dodatkowo metoda po dodaniu inkrementuje liczbę neuronów w warstwie w atrybucie
            num neurons.
        """
        self.neurons.append(neuron)
        self.num_neurons += 1

    def replace_oldest(self, candidate: OutputNeuron) -> None:
        """
            Metoda służy do zastąpywania najstaszego neuronu w warstwie
            przez nowo utworzonego kandydata, gdy liczba neuronów w warstwie
            jest maksymalna.
        """
        oldest = min(self.neurons, key=lambda n: n.addition_time)
        self.neurons.remove(oldest)
        self.neurons.append(candidate)

    def reset_psp(self) -> None:
        """
            Metoda zerująca potencjał post-synaptyczny wszystkich neuronów 
            w warstwie.
        """
        for neuron in self.neurons:
            neuron.psp = 0
