from oesnn_ad import OeSNN_AD
from neuron import Output_Neuron
import numpy as np

WINDOW = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test__fires_first_with_none():
    oesnn_ad = OeSNN_AD(WINDOW, 14, 10, 10, 0.5, 0.5, 0.5, 0.5)

    assert oesnn_ad._fires_first() is None


def test__fires_first():
    oesnn_ad = OeSNN_AD(WINDOW, 14, 3, 3, 0.5, 0.5, 0.5, 0.5)

    oesnn_ad.input_layer.orders = np.array([2, 1, 0])

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
