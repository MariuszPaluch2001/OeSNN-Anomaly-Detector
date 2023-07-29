"""
    Główny moduł projektu, w którym testowana jest efektywność modelu.
    
    Funkcje z tego modułu nie powinny być importowane.
"""

import csv
from typing import List, Tuple, Dict
import numpy as np

from oesnn_ad import OeSNNAD
from utils import get_data_from_path, get_all_files_paths, perf_measure, read_parameters


def parameters_tuning(stream: np.ndarray,
                      labels: List[bool], parameters: Dict) -> Tuple[Dict, float, float, float]:
    """
        Funkcja implementująca algorytm grid-search, który jest wykorzystywany do 
        znalezienia najlepszychhiperparametrów oesnn-ad dla danego strumienia danych.
        
        Jako argumenty funkcja przyjmuje:\n
            1. stream - strumień danych w postaci wektora najlepiej z biblioteki numpy
            2. labels - listę etykiet mówiących o próbkach czy są anomaliami
            3. parameters - słownik, w którym klucze to hiperparametry, a wartości to tuple,
            które opisują zakres przeszukiwań i są w postaci (początek, koniec, krok)
        
        Funkcja zwraca krotkę z następującymi wartościami:\n
            1. słownik hiperparametr-wartośc dla najlepszego wyniku
            2. recall najlepszego wyniku
            3. precission najlepszego wyniku
            4. f1 najlepszego wyniku
    """
    best_parameters, best_recall, best_precission, best_f1 = None, 0.0, 0.0, 0.0
    for no_size in range(*parameters['NOsize']):
        for w_size in range(*parameters['Wsize']):
            for ni_size in range(*parameters['NIsize']):
                for ts_coef in range(*parameters['TS']):
                    for beta in np.arange(*parameters['Beta']):
                        for sim in np.arange(*parameters['sim']):
                            for mod in np.arange(*parameters['mod']):
                                for c_coef in np.arange(*parameters['C']):
                                    for ksi in np.arange(*parameters['ksi']):
                                        for epsilon in range(*parameters['epsilon']):
                                            oesnn_ad = OeSNNAD(stream,
                                                                window_size=w_size,
                                                                num_in_neurons=ni_size,
                                                                num_out_neurons=no_size,
                                                                ts_factor=ts_coef,
                                                                mod=mod,
                                                                c_factor=c_coef,
                                                                epsilon=epsilon,
                                                                ksi=ksi,
                                                                sim=sim,
                                                                beta=beta)
                                            detection_result = oesnn_ad.predict()
                                            recall, precission, f_1 = perf_measure(
                                                detection_result, labels)
                                            if f_1 > best_f1:
                                                best_f1 = f_1
                                                best_precission = precission
                                                best_recall = recall
                                                best_parameters = {
                                                    'NOsize': no_size,
                                                    'Wsize': w_size,
                                                    'NIsize': ni_size,
                                                    'Beta': beta,
                                                    'TS': ts_coef,
                                                    'sim': sim,
                                                    'mod': mod,
                                                    'C': c_coef,
                                                    'ksi': ksi,  # error factor
                                                    'epsilon': epsilon  # anomaly factor
                                                }

    return best_parameters, best_recall, best_precission, best_f1


def main() -> None:
    """
        Główna funkcja programu. Odczytuje zakres hiperparametrów z plików json, odczytuje
        strumienie z plików csv, uruchamia strojenie hiperparametrów, zapisuje najlepsze wyniki
        do pliku csv dla danego datasetu.
        
        Nie przyjmuje żadnych parametrów, i niczego nie zwraca.
    """
    parameters_nab = read_parameters("parameters_NAB.json")
    parameters_yahoo = read_parameters("parameters_Yahoo.json")
    with open('nab_result.csv', 'w', encoding='UTF8') as file:
        writer = csv.writer(file)

        writer.writerow(("dataset", "filename", "f1", "recall",
                        "precission") + tuple(parameters_nab.keys()))
        for path in get_all_files_paths("../data/NAB"):
            dataset = path.split("/")[-2]
            filename = path.split("/")[-1][:-4]
            print(f"dataset={dataset}, filename={filename}")
            data, labels = get_data_from_path(path, True)
            param, recall, precission, f_1 = parameters_tuning(
                data, labels, parameters_nab)
            if param is not None:
                writer.writerow((dataset, filename, f_1, recall,
                                precission) + tuple(param.values()))
            else:
                writer.writerow((dataset, filename, f_1, recall,
                                precission) + (None, ) * 10)
    with open('yahoo_result.csv', 'w', encoding='UTF8') as file:
        writer = csv.writer(file)

        writer.writerow(("dataset", "filename", "f1", "recall",
                        "precission") + tuple(parameters_yahoo.keys()))
        for path in get_all_files_paths("../data/Yahoo"):
            dataset = path.split("/")[-2]
            filename = path.split("/")[-1][:-4]
            print(f"dataset={dataset}, filename={filename}")
            data, labels = get_data_from_path(path, False)
            param, recall, precission, f_1 = parameters_tuning(
                data, labels, parameters_yahoo)
            if param is not None:
                writer.writerow((dataset, filename, f_1, recall,
                                precission) + tuple(param.values()))
            else:
                writer.writerow((dataset, filename, f_1, recall,
                                precission) + (None, ) * 10)


if __name__ == "__main__":
    main()
