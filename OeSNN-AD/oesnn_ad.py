from layer import Input_Layer, Output_Layer
from neuron import Output_Neuron, Input_Neuron

import numpy as np

from typing import List, Generator


class OeSNN_AD:

    def __init__(self, stream: np.ndarray, window_size: int,
                 num_in_neurons: int, num_out_neurons: int,
                 TS: float, mod: float, C: float, epsilon: float,
                 ksi: float = 0.0, sim: float = 0.0) -> None:

        self.stream = stream
        self.stream_len = self.stream.shape[0]
        self.window_size = window_size

        self.input_layer: Input_Layer = Input_Layer(num_in_neurons)
        self.output_layer: Output_Layer = Output_Layer(num_out_neurons)

        self.TS = TS
        self.mod = mod
        self.C = C

        self.gamma = self.C * \
            (1 - self.mod**(2*num_in_neurons)) / (1 - self.mod**2)
        self.epsilon = epsilon
        self.ksi = ksi
        self.sim = sim

        self.values: List[float] = []
        self.anomalies: List[bool] = []
        self.errors: List[float] = []

    def _get_window_from_stream(self, begin_idx: int, end_idx: int) -> np.ndarray:
        return self.stream[begin_idx: end_idx]

    def _init_new_arrays_for_predict(self, window: np.ndarray) -> None:
        self.values = np.random.normal(
            np.mean(window), np.std(window), self.window_size).tolist()
        self.errors = [np.abs(xt - yt) for xt, yt in zip(window, self.values)]
        self.anomalies = [False] * self.window_size

    def predict(self) -> np.ndarray:
        window = self._get_window_from_stream(0, self.window_size)

        self._init_new_arrays_for_predict(window)
        for t in range(self.window_size + 1, self.stream_len):
            self.input_layer.set_orders(window, self.TS, self.mod)

            window = self._get_window_from_stream(t - self.window_size, t)

            self._anomaly_detection(window)

            self._learning(window, t)

        return np.array(self.anomalies)

    def _anomaly_detection(self, window: np.ndarray) -> None:
        xt = window[-1]
        first_fired_neuron = self._fires_first()
        if first_fired_neuron is None:
            self.values.append(None)
            self.errors.append(np.inf)
            self.anomalies.append(True)
        else:
            self.values.append(first_fired_neuron.output_value)
            self.errors.append(
                np.abs(xt - first_fired_neuron.output_value))
            self.anomalies.append(self._anomaly_classification())

    def _anomaly_classification(self) -> bool:
        error_t = self.errors[-1]
        errors_window = self.errors[-(self.window_size):-1]
        anomalies_window = self.anomalies[-(self.window_size - 1):]

        errors_for_non_anomalies = [err for err, classification in zip(
            errors_window, anomalies_window) if not classification]

        return not (
            (not errors_for_non_anomalies) or (error_t - np.mean(errors_for_non_anomalies)
                                               < np.std(errors_for_non_anomalies) * self.epsilon)
        )

    def _learning(self, window: np.ndarray, neuron_age: int) -> None:
        anomaly_t, xt = self.anomalies[-1], window[-1]
        candidate_neuron = self.output_layer.make_candidate(window, self.input_layer.orders,
                                                            self.mod, self.C, neuron_age)

        if not anomaly_t:
            candidate_neuron.output_value += (xt -
                                              candidate_neuron.output_value) * self.ksi

        most_familiar_neuron, dist = self.output_layer.find_most_similar(
            candidate_neuron)

        if dist <= self.sim:
            most_familiar_neuron.update_neuron(candidate_neuron)
        elif self.output_layer.num_neurons < self.output_layer.max_outpt_size:
            self.output_layer.add_new_neuron(candidate_neuron)
        else:
            self.output_layer.replace_oldest(candidate_neuron)

    def _update_psp(self, neuron_input: Input_Neuron) -> Generator[Output_Neuron, None, None]:
        for n_out in self.output_layer:
            n_out.PSP += n_out.weights[neuron_input.id] * \
                (self.mod ** neuron_input.order)

            if n_out.PSP > self.gamma:
                yield n_out

    def _fires_first(self) -> Output_Neuron | None:
        self.output_layer.reset_psp()

        for neuron_input in self.input_layer:
            to_fire = [n_out for n_out in self._update_psp(neuron_input)]

            if to_fire:
                return max(to_fire, key=lambda x: x.PSP)
