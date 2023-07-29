"""
    Moduł testujący wybrane funkcje z modułu utils.
"""

import numpy as np
from utils import convert_numpy_array_int_to_booleans, perf_measure


def test_convert_numpy_array_int_to_booleans():
    """
        Test sprawdzający czy wektor intów jest dobrze konwertowany na wektor booli.
    """
    array = np.array([1, 0, 0, 1, 0])

    np.testing.assert_array_equal(convert_numpy_array_int_to_booleans(
        array), np.array([True, False, False, True, False]))


def test_perf_measure():
    """
        Test sprawdzający czy funkcja perf_measure dobrze liczy miary jakości.
    """
    preds, correct = [True, True, False, False], [False, True, False, True]
    recall, precission, f_1 = perf_measure(preds, correct)
    assert recall == 0.5
    assert precission == 0.5
    assert f_1 == 0.5
