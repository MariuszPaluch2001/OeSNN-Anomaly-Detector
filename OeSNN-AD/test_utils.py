"""
    Module tests utils functions.
"""

import numpy as np

from utils import perf_measure

def test_perf_measure():
    """
        Test assert if method correctly calculate performance measures.
    """
    preds, correct = [True, True, False, False], [False, True, False, True]
    recall, precission, f_1 = perf_measure(preds, correct)
    assert recall == 0.5
    assert precission == 0.5
    assert f_1 == 0.5
