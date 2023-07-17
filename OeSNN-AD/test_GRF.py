from grf_init import GRF_Init
import numpy as np

WINDOW = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test__get_width_vec():
    grf = GRF_Init(WINDOW, 7, 1, 0.5)

    result = grf._get_width_vec()
    correct = np.repeat(0.18, 7)
    np.testing.assert_array_almost_equal(result, correct, decimal=2)


def test__get_center_vec():
    grf = GRF_Init(WINDOW, 7, 1, 0.5)

    result = grf._get_center_vec()
    correct = np.array([-0.17,  0.01,  0.19,  0.37,  0.55,  0.73,  0.91])
    np.testing.assert_array_almost_equal(result, correct, decimal=2)


def test__get_excitation():
    grf = GRF_Init(WINDOW, 7, 1, 0.5)
    width_v = np.repeat(0.18, 7)
    center_v = np.array([-0.17,  0.01,  0.19,  0.37,  0.55,  0.73,  0.91])

    result = grf._get_excitation(width_v, center_v)
    correct = np.array([0.001, 0.024, 0.227, 0.770, 0.962, 0.442, 0.074])
    np.testing.assert_array_almost_equal(result, correct, decimal=3)


def test__get_firing_time():
    grf = GRF_Init(WINDOW, 7, 1, 0.5)
    excitation = np.array([0.001, 0.024, 0.227, 0.770, 0.962, 0.442, 0.074])

    result = grf._get_firing_time(excitation)
    correct = np.array([0.999, 0.976, 0.773, 0.230, 0.038, 0.558, 0.926])
    np.testing.assert_array_almost_equal(result, correct, decimal=3)


def test__get_order():
    grf = GRF_Init(WINDOW, 7, 1, 0.5)
    firing_time = np.array([0.999, 0.976, 0.773, 0.230, 0.038, 0.558, 0.926])

    result = grf._get_order(firing_time)
    correct = np.array([6, 5, 3, 1, 0, 2, 4])
    np.testing.assert_array_equal(result, correct)


def test_get_order():
    grf = GRF_Init(WINDOW, 7, 1, 0.5)

    result = grf.get_order()
    correct = np.array([6, 5, 3, 1, 0, 2, 4])
    np.testing.assert_array_equal(result, correct)

# def test__get_weights():
#     grf = GRF_Init(WINDOW, 7, 10, 1, 0.5)
#     orders = np.array([6,5,3,1,0,2,4])

#     result = grf._get_weights(orders)
#     correct = np.array([np.repeat(0.015625, 10),
#                        np.repeat(0.03125, 10),
#                        np.repeat(0.125, 10),
#                        np.repeat(0.5, 10),
#                        np.repeat(1, 10),
#                        np.repeat(0.25, 10),
#                        np.repeat(0.0625, 10)])
#     np.testing.assert_array_almost_equal(result, correct, decimal = 3)
