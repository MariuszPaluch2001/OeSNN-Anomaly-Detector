"""
    Module contains utils function which cannot be placed in other modules.
    These functions are responsible for reading data from files, and function to 
    calc performance of model.
"""
from typing import List, Dict, Tuple, Generator

import os
import json
import numpy as np
import numpy.typing as npt

def get_all_files_paths(data_root_folder_name: str) -> Generator[str, None, None]:
    """        
        Function return paths of all datasts files in repository tree.
        
        Args:
            data_root_folder_name (str): path to data root folder

        Yields:
            Generator[str, None, None]: yields dataset files paths
    """
    for path, _, files in os.walk(data_root_folder_name):
        for name in files:
            yield os.path.join(path, name)

def get_data_from_path(path: str,
                       is_nab: bool) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """
        Function read datastream and labels from datastream.

        Args:
            path (str): path to dataset file
            is_nab (bool): flag which tell if dataset is from NAB repository

        Returns:
            Tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]: First element of tuple is
            datastream and second one are labels
    """
    if is_nab:
        data = np.loadtxt(path, delimiter=",", dtype=float,
                          skiprows=1, usecols=range(1, 3))
    else:
        data = np.loadtxt(path, delimiter=",",
                          dtype=float, usecols=range(1, 3))

    return data[:, 0], data[:, 1]

def read_parameters(path: str) -> Dict:
    """
        Function to read hyperparameters from json file.

        Args:
            path (str): path to json file

        Returns:
            Dict: dictionary with hyperparameters
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def convert_numpy_array_int_to_booleans(array: npt.NDArray[np.intp]) -> npt.NDArray[np.bool_]:
    """
        Function convert numpy int array to numpy boolean array.

        Args:
            array (npt.NDArray[np.intp]): numpy array with int values

        Returns:
            npt.NDArray[np.bool_]: numpy array with boolean values
    """
    return array.astype(bool)

def perf_measure(y_hat: List, y_actual: List) -> Tuple[float, float, float]:
    """
        Function to calculate performance measure such as: recall, precission and f1.
        In such order they are returned from function.
        
        Args:
            y_hat (List): list of prediction 
            y_actual (List): list of true values

        Returns:
            Tuple[float, float, float]: recall, precission, f1 measures
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
