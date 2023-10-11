import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PyEMD import CEEMDAN

from oesnn_ad import OeSNNAD


class DecomposeOeSNNAD:
    
    def __init__(self, stream: npt.NDArray[np.float64],
                 range_threshold= 1e-3,
                 total_power_threshold= 1e-2) -> None:
        self.stream = stream
        self.ceemdan = CEEMDAN(range_thr=range_threshold, total_power_thr=total_power_threshold)
    
    def _decompose(self, channels: int):
        return self.ceemdan.ceemdan(self.stream, max_imf=channels)
        
    def predict(self, tuning_type: str, cemmndan_channels: int=-1):
        c_imfs = self._decompose(cemmndan_channels)
        results = [OeSNNAD(component, window_size=4).predict() for component in c_imfs]
        print(results)
        return results
            

if __name__ == "__main__":
    x = np.array([1,2,4,6,7,63,23,12,342,5,63,21])
    res = DecomposeOeSNNAD(x).predict('whatever')