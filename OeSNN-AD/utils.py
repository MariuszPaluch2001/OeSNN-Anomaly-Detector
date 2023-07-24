import os
import json
import numpy as np
from typing import List, Tuple, Generator, Dict


def get_all_files_paths(data_root_folder_name: str) -> Generator[str, None, None]:
    for path, _, files in os.walk(data_root_folder_name):
        for name in files:
            yield os.path.join(path, name)


def get_data_from_path(filename: str, is_nab: bool) -> Tuple[np.ndarray, np.ndarray]:
    if is_nab:
        data = np.loadtxt(filename, delimiter=",", dtype=float,
                          skiprows=1, usecols=range(1, 3))
    else:
        data = np.loadtxt(filename, delimiter=",",
                          dtype=float, usecols=range(1, 3))

    return data[:, 0], data[:, 1]


def read_parameters(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def convert_numpy_array_int_to_booleans(array: np.ndarray[int]) -> np.ndarray[bool]:
    return array.astype(bool)


def perf_measure(y_hat: List, y_actual: List) -> Tuple[float, float, float]:
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for y_h, y_act in zip(y_hat, y_actual):
        if y_h and y_act:
            TP += 1
        if y_h and not y_act:
            FP += 1
        if not y_h and not y_act:
            TN += 1
        if not y_h and y_act:
            FN += 1

    if TP or FP:
        precission = TP / (TP + FP)
    else:
        precission = 0
    if TP or FN:
        recall = TP / (TP + FN)
    else:
        recall = 0
    if recall or precission:
        f1 = (2 * precission * recall) / (precission + recall)
    else:
        f1 = 0
    return recall, precission, f1
