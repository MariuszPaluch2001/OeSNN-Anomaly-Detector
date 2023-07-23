from utils import get_data_from_path, get_all_files_paths, perf_measure
from oesnn_ad import OeSNN_AD
import csv
from typing import List, Tuple, Dict
import numpy as np

# (begin, end, step)
NAB_parameters = {
    'NOsize': (50, 51, 100),
    'Wsize': (100, 600, 100),
    'NIsize': (10, 11, 20),
    'Beta': (1.6, 1.7, 0.2),
    'TS': (1000, 1001, 1000),
    'sim': (0.15, 0.16, 0.15),
    'mod': (0.6, 0.61, 0.1),
    'C': (0.6, 0.61, 0.1),
    'ksi': (0.9, 0.91, 0.1),  # error factor
    'epsilon': (2, 7, 1)  # anomaly factor
}
# (begin, end, step)
Yahoo_parameters = {
    'NOsize': (50, 51, 100),
    'Wsize': (50, 400, 50),
    'NIsize': (50, 400, 50),
    'Beta': (1.6, 1.61, 0.2),
    'TS': (1000, 1001, 1000),
    'sim': (0.15, 0.151, 0.15),
    'mod': (0.6, 0.61, 0.1),
    'C': (0.8, 0.81, 0.1),
    'ksi': (0.9, 0.91, 0.1),  # error factor
    'epsilon': (5, 17, 1)  # anomaly factor
}


def parameters_tuning(stream: np.ndarray, labels: List[bool], parameters: Dict) -> Tuple[Dict, float, float, float]:
    best_parameters, best_recall, best_precission, best_f1 = None, 0.0, 0.0, 0.0
    for no_size in range(*parameters['NOsize']):
        for w_size in range(*parameters['Wsize']):
            for ni_size in range(*parameters['NIsize']):
                for TS in range(*parameters['TS']):
                    for beta in np.arange(*parameters['Beta']):
                        for sim in np.arange(*parameters['sim']):
                            for mod in np.arange(*parameters['mod']):
                                for C in np.arange(*parameters['C']):
                                    for ksi in np.arange(*parameters['ksi']):
                                        for epsilon in range(*parameters['epsilon']):
                                            oesnn_ad = OeSNN_AD(stream, window_size=w_size, num_in_neurons=ni_size, num_out_neurons=no_size,
                                                                TS=TS, mod=mod, C=C, epsilon=epsilon, ksi=ksi, sim=sim)
                                            detection_result = oesnn_ad.predict()
                                            recall, precission, f1 = perf_measure(
                                                detection_result, labels)
                                            if f1 > best_f1:
                                                best_f1 = f1
                                                best_precission = precission
                                                best_recall = recall
                                                best_parameters = {
                                                    'NOsize': no_size,
                                                    'Wsize': w_size,
                                                    'NIsize': ni_size,
                                                    'Beta': beta,
                                                    'TS': TS,
                                                    'sim': sim,
                                                    'mod': mod,
                                                    'C': C,
                                                    'ksi': ksi,  # error factor
                                                    'epsilon': epsilon  # anomaly factor
                                                }

    return best_parameters, best_recall, best_precission,  best_f1


def main() -> None:
    with open('nab_result.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        writer.writerow(("dataset", "filename", "f1", "recall",
                        "precission") + tuple(NAB_parameters.keys()))
        for path in get_all_files_paths("../data/NAB"):
            dataset = path.split("/")[-2]
            filename = path.split("/")[-1][:-4]
            data, labels = get_data_from_path(path, True)
            param, recall, precission, f1 = parameters_tuning(
                data, labels, NAB_parameters)
            if param is not None:
                writer.writerow((dataset, filename, f1, recall,
                                precission) + tuple(param.values()))
            else:
                writer.writerow((dataset, filename, f1, recall,
                                precission) + (None, ) * 10)
    with open('yahoo_result.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        writer.writerow(("dataset", "filename", "f1", "recall",
                        "precission") + tuple(Yahoo_parameters.keys()))
        for path in get_all_files_paths("../data/Yahoo"):
            dataset = path.split("/")[-2]
            filename = path.split("/")[-1][:-4]
            data, labels = get_data_from_path(path, False)
            param, recall, precission,  f1 = parameters_tuning(
                data, labels, Yahoo_parameters)
            if param is not None:
                writer.writerow((dataset, filename, f1, recall,
                                precission) + tuple(param.values()))
            else:
                writer.writerow((dataset, filename, f1, recall,
                                precission) + (None, ) * 10)


if __name__ == "__main__":
    main()
