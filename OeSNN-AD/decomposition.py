from oesnn_ad import OeSNNAD
from PyEMD import CEEMDAN
import numpy.typing as npt
import numpy as np

class DecomposeOeSNNAD:
    
    def __init__(self, stream: npt.NDArray[np.float64]) -> None:
        self.ceemdan = CEEMDAN(range_thr=0.001, total_power_thr=0.01)
        self.c_imfs = self.ceemdan(stream, max_imf=10)
        print(self.c_imfs)
    
    def _decompose(self):
        ...
        
    def predict(self):
        ...

if __name__ == "__main__":
    import matplotlib.pyplot as plt        
    x = np.array([1,2,4,6,7,63,23,12,342,5,63,21])
    res = DecomposeOeSNNAD(np.array([1,2,4,6,7,63,23,12,342,5,63,21])).c_imfs
    print(res.transpose())
    plt.plot(x)
    plt.plot(res.transpose())
    plt.savefig(f"output.png")
    plt.close()