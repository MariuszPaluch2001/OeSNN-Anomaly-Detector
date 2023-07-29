"""
    Moduł zawiera pomocnicze funkcje, które nie mogły znaleźć się w innych modułach.
    Głównie znajdują się tutaj funkcje do odczytywania danych z plików, oraz funkcja
    perf_measure do zliczania miar jakości.
"""
from typing import List, Dict, Tuple, Generator

import os
import json
import numpy as np


def get_all_files_paths(data_root_folder_name: str) -> Generator[str, None, None]:
    """
        Funkcja zwraca ścieżki do wszystkich plików w strukturze folderu, począwszy
        od folderu-korzenia, który jest podany jako argument funkcji (data_root_folder_name).
        
        Funkcja ta jest generatorem.
    """
    for path, _, files in os.walk(data_root_folder_name):
        for name in files:
            yield os.path.join(path, name)


def get_data_from_path(filename: str, is_nab: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
        Funkcja ta odczytuje strumień danych z pliku, i zwraca go wraz z odpowiadającymi mu
        etykietami.
        
        Funkcja ta przyjmuje nazwę pliku w postaci stringa, oraz flagę is_nab, mówiącą czy
        jest to zbiór nab. W przyszłości jak ujednolice zbiory danych to do wywalenia.
    """
    if is_nab:
        data = np.loadtxt(filename, delimiter=",", dtype=float,
                          skiprows=1, usecols=range(1, 3))
    else:
        data = np.loadtxt(filename, delimiter=",",
                          dtype=float, usecols=range(1, 3))

    return data[:, 0], data[:, 1]


def read_parameters(path: str) -> Dict:
    """
        Funkcja do otczytywania hiperparametrów z pliku json.
        
        Jako argument funkcja przyjmuje ścieżkę do pliku w postaci stringa.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def convert_numpy_array_int_to_booleans(array: np.ndarray[int]) -> np.ndarray[bool]:
    """
        Funkcja do konwersji numpy listy z intów do wartości bool.
    """
    return array.astype(bool)


def perf_measure(y_hat: List, y_actual: List) -> Tuple[float, float, float]:
    """
        Funkcja służy do liczenia miar jakości takich jak: recall, precission i f1.
        W takiej kolejności też te wartości są zwracane.
        
        Jako argumenty funkcja przyjmuje:\n
            1. y_hat - listę z wartościami stworzonymi przez model
            2. y_actual - listę z prawdziwymi wartościami
    """
    true_positives = 0
    false_positives = 0
    true_negative = 0
    false_negative = 0
    for y_h, y_act in zip(y_hat, y_actual):
        if y_h and y_act:
            true_positives += 1
        if y_h and not y_act:
            false_positives += 1
        if not y_h and not y_act:
            true_negative += 1
        if not y_h and y_act:
            false_negative += 1

    if true_positives or false_positives:
        precission = true_positives / (true_positives + false_positives)
    else:
        precission = 0
    if true_positives or false_negative:
        recall = true_positives / (true_positives + false_negative)
    else:
        recall = 0
    if recall or precission:
        f_1 = (2 * precission * recall) / (precission + recall)
    else:
        f_1 = 0
    return recall, precission, f_1
