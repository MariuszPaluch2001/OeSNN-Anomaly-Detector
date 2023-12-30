from threading import Thread

import numpy as np
import numpy.typing as npt
from PyEMD import CEEMDAN

from oesnn_ad import OeSNNAD


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

    def _merge(self, detections: npt.NDArray) -> npt.NDArray:
        summed_columns = detections.sum(axis=0)
        return (summed_columns / detections.shape[0]) > 0.5

    def _merge_geometric_seq(self, detections: npt.NDArray, common_ratio: float):
        if common_ratio < 0.5 or common_ratio > 1.0:
            raise ValueError("Common ratio should be in range [0.5, 1.0]")

        weights = np.array(
            [common_ratio**n for n in range(0, detections.shape[0])])
        weighted_detections = (
            detections.T * weights).T
        summed_columns = weighted_detections.sum(axis=0)
        return summed_columns > weights.sum() / 2

    def predict(self, cemmndan_channels: int = -1, merge_type: str = "committee", common_ratio=None, **parameters):
        channels = list(self._decompose(cemmndan_channels))
        channels.insert(0, self.stream)
        detections = []
        threads = [Thread(
            target=detections.append(
                (idx, OeSNNAD(component, **parameters).predict()))
        ) for idx, component in enumerate(channels)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        detections = [x[1] for x in sorted(detections)]
        if (merge_type == "committee"):
            return self._merge(np.array(detections))
        elif (merge_type == "geometric"):
            return self._merge_geometric_seq(np.array(detections), common_ratio)
        else:
            raise ValueError("Incorrect merge type")
