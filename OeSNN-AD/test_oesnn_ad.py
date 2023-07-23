from pytest import approx
from oesnn_ad import OeSNN_AD
from neuron import Output_Neuron, Input_Neuron
import numpy as np

WINDOW = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test__anomaly_classification_without_correct_values():
    oesnn_ad = OeSNN_AD(WINDOW, 5, 3, 3, 0.5, 0.5, 0.5, 0.5)
    oesnn_ad.errors = [0.0 for _ in range(5)]
    oesnn_ad.anomalies = [True for _ in range(4)]
    assert not oesnn_ad._anomaly_classification()


def test__anomaly_classification_with_anomaly_classified():
    oesnn_ad = OeSNN_AD(WINDOW, 5, 3, 3, 0.5, 0.5, 0.5, 0.5)
    oesnn_ad.errors = [0.1, 0.2, 0.15, 0.1, 0.9]
    oesnn_ad.anomalies = [False for _ in range(4)]
    assert oesnn_ad._anomaly_classification()


def test__anomaly_classification_with_not_anomaly_classified():
    oesnn_ad = OeSNN_AD(WINDOW, 5, 3, 3, 0.5, 0.5, 0.5, 0.5)
    oesnn_ad.errors = [0.1, 0.2, 0.15, 0.1, 0.16]
    oesnn_ad.values = [False for _ in range(4)]
    assert not oesnn_ad._anomaly_classification()


def test__get_window_from_stream():
    oesnn_ad = OeSNN_AD(WINDOW, 5, 3, 3, 0.5, 0.5, 0.5, 0.5)
    result = oesnn_ad._get_window_from_stream(0, 5)
    correct = np.array([0.5, 0.3, 0.4, 0.3, 0.6])
    np.testing.assert_array_equal(result, correct)


def test__init_new_arrays_for_predict():
    oesnn_ad = OeSNN_AD(WINDOW, 5, 3, 3, 0.5, 0.5, 0.5, 0.5)
    assert len(oesnn_ad.values) == 0
    assert len(oesnn_ad.anomalies) == 0
    assert len(oesnn_ad.errors) == 0

    oesnn_ad._init_new_arrays_for_predict(WINDOW[0:5])
    np.testing.assert_array_equal(oesnn_ad.anomalies, [False] * 5)
    assert len(oesnn_ad.values) == 5
    assert len(oesnn_ad.errors) == 5


def test_predict():
    assert True


def test__anomaly_detection_without_firing():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=5, num_in_neurons=1,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)
    oesnn_ad._anomaly_detection(np.array([1, 2, 3]))
    assert oesnn_ad.values[0] is None
    assert np.isinf(oesnn_ad.errors[0])
    assert oesnn_ad.anomalies[0]


def test__anomaly_detection_with_firing():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=5, num_in_neurons=1,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)
    oesnn_ad.errors = [0.99, 0.15, 0.99, 0.07]
    oesnn_ad.anomalies = [True, False, True, False]

    neuron_input1 = Input_Neuron(firing_time=0.5, id=0, order=2)
    neuron_input2 = Input_Neuron(firing_time=0.5, id=1, order=1)
    neuron_input3 = Input_Neuron(firing_time=0.5, id=2, order=0)

    oesnn_ad.input_layer.neurons = [
        neuron_input1, neuron_input2, neuron_input3]

    neuron_output1 = Output_Neuron(weights=np.array(
        [0.1, 0.4, 0.7]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output2 = Output_Neuron(weights=np.array(
        [0.2, 0.1, 0.8]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output3 = Output_Neuron(weights=np.array(
        [0.3, 0.8, 0.9]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)

    oesnn_ad.output_layer.add_new_neuron(neuron_output1)
    oesnn_ad.output_layer.add_new_neuron(neuron_output2)
    oesnn_ad.output_layer.add_new_neuron(neuron_output3)
    oesnn_ad._anomaly_detection(np.array([1.0, 0.55, 1.01, 0.57, 0.6]))
    assert oesnn_ad.values[-1] == 0.5
    assert oesnn_ad.errors[-1] == approx(0.1, abs=1e-1)
    assert not oesnn_ad.anomalies[-1]


def test__anomaly_classification_without_non_anomaly_in_window():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=5, num_in_neurons=1,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)
    oesnn_ad.errors = [0.99, 0.99, 0.99, 0.99, 0.99]
    oesnn_ad.anomalies = [True, True, True, True]

    assert not oesnn_ad._anomaly_classification()


def test__anomaly_classification_without_anomaly_result():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=5, num_in_neurons=1,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)
    oesnn_ad.errors = [0.99, 0.15, 0.99, 0.07, 0.09]
    oesnn_ad.anomalies = [True, False, True, False]
    assert not oesnn_ad._anomaly_classification()


def test__anomaly_classification_with_anomaly_result():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=5, num_in_neurons=1,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)
    oesnn_ad.errors = [0.99, 0.15, 0.99, 0.07, 0.68]
    oesnn_ad.anomalies = [True, False, True, False]
    assert oesnn_ad._anomaly_classification()


def test__learning_with_update():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=5, num_in_neurons=3,
                        num_out_neurons=5, TS=0.5, mod=0.3, C=1.0, epsilon=0.5, sim=1.0)

    oesnn_ad.anomalies.append(True)

    neuron_input1 = Input_Neuron(firing_time=0.5, id=0, order=2)
    neuron_input2 = Input_Neuron(firing_time=0.5, id=1, order=1)
    neuron_input3 = Input_Neuron(firing_time=0.5, id=2, order=0)

    oesnn_ad.input_layer.neurons = [
        neuron_input1, neuron_input2, neuron_input3]

    neuron_output1 = Output_Neuron(weights=np.array(
        [0.1, 0.4, 0.7]), gamma=0.5, output_value=0.5, M=0, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output2 = Output_Neuron(weights=np.array(
        [0.2, 0.1, 0.8]), gamma=0.5, output_value=0.5, M=0, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output3 = Output_Neuron(weights=np.array(
        [0.3, 0.8, 0.9]), gamma=0.5, output_value=0.5, M=0, addition_time=0.5, PSP=0.0, max_PSP=2)

    oesnn_ad.output_layer.add_new_neuron(neuron_output1)
    oesnn_ad.output_layer.add_new_neuron(neuron_output2)
    oesnn_ad.output_layer.add_new_neuron(neuron_output3)

    oesnn_ad._learning(np.array([1, 2, 3]), 1)

    assert len(oesnn_ad.output_layer.neurons) == 3
    assert 1 in [n.M for n in oesnn_ad.output_layer]


def test__learning_with_add_new_neuron():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=5, num_in_neurons=5,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5, sim=1.0)

    oesnn_ad.anomalies.append(True)
    assert len(oesnn_ad.output_layer.neurons) == 0
    oesnn_ad._learning(np.array([1, 2, 3]), 1)
    assert len(oesnn_ad.output_layer.neurons) == 1


def test__learning_with_replace_oldest():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=5, num_in_neurons=3,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5, sim=0.1)

    oesnn_ad.anomalies.append(True)

    neuron_input1 = Input_Neuron(firing_time=0.5, id=0, order=2)
    neuron_input2 = Input_Neuron(firing_time=0.5, id=1, order=1)
    neuron_input3 = Input_Neuron(firing_time=0.5, id=2, order=0)

    oesnn_ad.input_layer.neurons = [
        neuron_input1, neuron_input2, neuron_input3]

    neuron_output1 = Output_Neuron(weights=np.array(
        [0.1, 0.4, 0.7]), gamma=0.5, output_value=0.5, M=0, addition_time=1, PSP=0.0, max_PSP=2)
    neuron_output2 = Output_Neuron(weights=np.array(
        [0.2, 0.1, 0.8]), gamma=0.5, output_value=0.5, M=0, addition_time=5, PSP=0.0, max_PSP=2)
    neuron_output3 = Output_Neuron(weights=np.array(
        [0.3, 0.8, 0.9]), gamma=0.5, output_value=0.5, M=0, addition_time=10, PSP=0.0, max_PSP=2)

    oesnn_ad.output_layer.add_new_neuron(neuron_output1)
    oesnn_ad.output_layer.add_new_neuron(neuron_output2)
    oesnn_ad.output_layer.add_new_neuron(neuron_output3)

    assert len(oesnn_ad.output_layer.neurons) == 3
    oesnn_ad._learning(np.array([1, 2, 3]), 15)
    assert len(oesnn_ad.output_layer.neurons) == 3
    assert 5 == min([n.addition_time for n in oesnn_ad.output_layer])
    assert 15 == max([n.addition_time for n in oesnn_ad.output_layer])


def test__update_psp_case_with_one_input_neuron():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=3, num_in_neurons=1,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)

    assert oesnn_ad.gamma == 1.0
    neuron_input = Input_Neuron(firing_time=0.5, id=0, order=0)
    neuron_output1 = Output_Neuron(weights=np.array(
        [0.1]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=1.0, max_PSP=2)
    neuron_output2 = Output_Neuron(weights=np.array(
        [0.2]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=1.0, max_PSP=2)
    neuron_output3 = Output_Neuron(weights=np.array(
        [0.3]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=1.0, max_PSP=2)

    oesnn_ad.output_layer.add_new_neuron(neuron_output1)
    oesnn_ad.output_layer.add_new_neuron(neuron_output2)
    oesnn_ad.output_layer.add_new_neuron(neuron_output3)

    updated_neurons = list(oesnn_ad._update_psp(neuron_input))
    assert len(updated_neurons) == 3
    assert neuron_output1.PSP == 1.1
    assert neuron_output2.PSP == 1.2
    assert neuron_output3.PSP == 1.3


def test__update_psp_case_with_multiple_input_neuron():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=3, num_in_neurons=3,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)

    assert oesnn_ad.gamma == 1.0981
    neuron_input1 = Input_Neuron(firing_time=0.5, id=0, order=2)
    neuron_input2 = Input_Neuron(firing_time=0.5, id=1, order=1)
    neuron_input3 = Input_Neuron(firing_time=0.5, id=2, order=0)

    neuron_output1 = Output_Neuron(weights=np.array(
        [0.1, 0.4, 0.7]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output2 = Output_Neuron(weights=np.array(
        [0.2, 0.1, 0.8]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output3 = Output_Neuron(weights=np.array(
        [0.3, 0.8, 0.9]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)

    oesnn_ad.output_layer.add_new_neuron(neuron_output1)
    oesnn_ad.output_layer.add_new_neuron(neuron_output2)
    oesnn_ad.output_layer.add_new_neuron(neuron_output3)

    updated_neurons = list(oesnn_ad._update_psp(neuron_input1))
    assert len(updated_neurons) == 0
    assert neuron_output1.PSP == approx(0.009, abs=1e-3)
    assert neuron_output2.PSP == approx(0.018, abs=1e-3)
    assert neuron_output3.PSP == approx(0.027, abs=1e-3)

    updated_neurons = list(oesnn_ad._update_psp(neuron_input2))
    assert len(updated_neurons) == 0
    assert neuron_output1.PSP == approx(0.129, abs=1e-3)
    assert neuron_output2.PSP == approx(0.048, abs=1e-3)
    assert neuron_output3.PSP == approx(0.267, abs=1e-3)

    updated_neurons = list(oesnn_ad._update_psp(neuron_input3))
    assert len(updated_neurons) == 1
    assert neuron_output1.PSP == approx(0.829, abs=1e-3)
    assert neuron_output2.PSP == approx(0.848, abs=1e-3)
    assert neuron_output3.PSP == approx(1.167, abs=1e-3)


def test__fires_first_with_none():
    oesnn_ad = OeSNN_AD(WINDOW, 14, 10, 10, 0.5, 0.5, 0.5, 0.5)

    assert oesnn_ad._fires_first() is None


def test__fires_first_with_one_input_neuron():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=3, num_in_neurons=1,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)

    neuron_input = Input_Neuron(firing_time=0.5, id=0, order=0)
    neuron_output1 = Output_Neuron(weights=np.array(
        [1.1]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output2 = Output_Neuron(weights=np.array(
        [1.2]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output3 = Output_Neuron(weights=np.array(
        [1.3]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)

    oesnn_ad.input_layer.neurons = [neuron_input]
    oesnn_ad.output_layer.add_new_neuron(neuron_output1)
    oesnn_ad.output_layer.add_new_neuron(neuron_output2)
    oesnn_ad.output_layer.add_new_neuron(neuron_output3)

    result = oesnn_ad._fires_first()

    assert result == neuron_output3
    assert result.PSP == 1.3


def test__fires_first_with_multiple_input_neuron():
    oesnn_ad = OeSNN_AD(stream=WINDOW, window_size=3, num_in_neurons=3,
                        num_out_neurons=3, TS=0.5, mod=0.3, C=1.0, epsilon=0.5)

    neuron_input1 = Input_Neuron(firing_time=0.5, id=0, order=2)
    neuron_input2 = Input_Neuron(firing_time=0.5, id=1, order=1)
    neuron_input3 = Input_Neuron(firing_time=0.5, id=2, order=0)

    oesnn_ad.input_layer.neurons = [
        neuron_input1, neuron_input2, neuron_input3]

    neuron_output1 = Output_Neuron(weights=np.array(
        [0.1, 0.4, 0.7]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output2 = Output_Neuron(weights=np.array(
        [0.2, 0.1, 0.8]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)
    neuron_output3 = Output_Neuron(weights=np.array(
        [0.3, 0.8, 0.9]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.0, max_PSP=2)

    oesnn_ad.output_layer.add_new_neuron(neuron_output1)
    oesnn_ad.output_layer.add_new_neuron(neuron_output2)
    oesnn_ad.output_layer.add_new_neuron(neuron_output3)

    result = oesnn_ad._fires_first()

    assert result == neuron_output3
    assert result.PSP == 1.167
