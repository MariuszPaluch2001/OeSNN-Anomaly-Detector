from utils import get_data_from_path, perf_measure
import numpy as np
import numpy.typing as npt
from PyEMD import CEEMDAN

from oesnn_ad import OeSNNAD
from threading import Thread


class DecomposeOeSNNAD:

    def __init__(self, stream: npt.NDArray[np.float64],
                 trials=100,
                 range_threshold=1e-3,
                 total_power_threshold=1e-2) -> None:
        self.stream = stream
        self.ceemdan = CEEMDAN(trials=trials, range_threshold=range_threshold,
                               total_power_threshold=total_power_threshold)

    def _decompose(self, channels: int):
        return self.ceemdan.ceemdan(self.stream, max_imf=channels)

    def _merge(self, streams: npt.NDArray) -> npt.NDArray:
        summed_columns = streams.sum(axis=0)
        return (summed_columns / streams.shape[0]) > 0.5

    def predict(self, cemmndan_channels: int = -1, **parameters):
        c_imfs = self._decompose(cemmndan_channels)
        streams = []
        threads = [Thread(
            target=streams.append(OeSNNAD(component, **parameters).predict())
        ) for component in c_imfs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return self._merge(np.array(streams))
