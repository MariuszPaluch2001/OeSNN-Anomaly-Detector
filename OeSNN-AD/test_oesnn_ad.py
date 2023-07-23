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


def test__anomaly_detection():
    assert True


def test__anomaly_classification():
    assert True


def test__learning():
    assert True


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
        [0.1, 0.4, 0.7]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.2, max_PSP=2)
    neuron_output2 = Output_Neuron(weights=np.array(
        [0.2, 0.1, 0.8]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.2, max_PSP=2)
    neuron_output3 = Output_Neuron(weights=np.array(
        [0.3, 0.8, 0.7]), gamma=0.5, output_value=0.5, M=0.5, addition_time=0.5, PSP=0.2, max_PSP=2)

    oesnn_ad.output_layer.add_new_neuron(neuron_output1)
    oesnn_ad.output_layer.add_new_neuron(neuron_output2)
    oesnn_ad.output_layer.add_new_neuron(neuron_output3)
    
    updated_neurons = list(oesnn_ad._update_psp(neuron_input1))
    assert len(updated_neurons) == 0
    assert neuron_output1.PSP == approx(0.209, abs=1e-3)
    assert neuron_output2.PSP == approx(0.218, abs=1e-3)
    assert neuron_output3.PSP == approx(0.227, abs=1e-3)
    
    updated_neurons = list(oesnn_ad._update_psp(neuron_input2))
    assert len(updated_neurons) == 0
    assert neuron_output1.PSP == approx(0.329, abs=1e-3)
    assert neuron_output2.PSP == approx(0.248, abs=1e-3)
    assert neuron_output3.PSP == approx(0.467, abs=1e-3)

    updated_neurons = list(oesnn_ad._update_psp(neuron_input3))
    assert len(updated_neurons) == 1
    assert neuron_output1.PSP == approx(1.029, abs=1e-3)
    assert neuron_output2.PSP == approx(1.048, abs=1e-3)
    assert neuron_output3.PSP == approx(1.167, abs=1e-3)


def test__fires_first_with_none():
    oesnn_ad = OeSNN_AD(WINDOW, 14, 10, 10, 0.5, 0.5, 0.5, 0.5)

    assert oesnn_ad._fires_first() is None


def test__fires_first():
    oesnn_ad = OeSNN_AD(WINDOW, 3, 3, 3, 0.5, 0.5, 0.5, 0.5)

    oesnn_ad.input_layer.neurons = [Input_Neuron(
        0.0, 2), Input_Neuron(0.0, 1), Input_Neuron(0.0, 0)]

    out_n1 = Output_Neuron(
        weights=np.array([0, 1, 2]),
        gamma=0.5,
        output_value=0.5,
        M=0.5,
        addition_time=0.5,
        PSP=0.5,
        max_PSP=1
    )
    out_n2 = Output_Neuron(
        weights=np.array([0, 1, 2]),
        gamma=0.5,
        output_value=0.5,
        M=0.5,
        addition_time=0.5,
        PSP=0.5,
        max_PSP=1
    )
    out_n3 = Output_Neuron(
        weights=np.array([0, 1, 2]),
        gamma=0.5,
        output_value=0.5,
        M=0.5,
        addition_time=0.5,
        PSP=0.5,
        max_PSP=1
    )

    oesnn_ad.output_layer.add_new_neuron(out_n1)
    oesnn_ad.output_layer.add_new_neuron(out_n2)
    oesnn_ad.output_layer.add_new_neuron(out_n3)

    result = oesnn_ad._fires_first()

    assert isinstance(result, Output_Neuron)
