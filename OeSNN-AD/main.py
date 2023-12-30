"""
    Main project's module which testing model effectiveness.
"""

import csv
import itertools
from typing import Dict, List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from oesnn_ad import OeSNNAD
from utils import (get_all_files_paths, get_data_from_path, perf_measure,
                   read_parameters)


def parameters_tuning(stream: npt.NDArray[np.float64],
                      labels: List[bool], parameters: Dict) -> Tuple[Dict, float, float, float]:
    """
        A function that implements the grid-search algorithm that is used for
        finding the best oesnn-ad hyperparameters for a given data stream.

        Args:
            stream (npt.NDArray[np.float64]): data stream's vector
            labels (List[bool]): labels vector
            parameters (Dict): _description_

        Returns:
            Tuple[Dict, float, float, float]: dict with best parameters, 
                                              recall for the best result,
                                              precission for the best result, 
                                              f1 for the best result
    """
    best_parameters, best_recall, best_precission, best_f1 = None, 0.0, 0.0, 0.0
    for no_size, w_size, ni_size, ts_coef, beta, sim, mod, c_coef, ksi, epsilon in itertools.product(
        np.arange(*parameters['NOsize']),
        np.arange(*parameters['Wsize']),
        np.arange(*parameters['NIsize']),
        np.arange(*parameters['TS']),
        np.arange(*parameters['Beta']),
        np.arange(*parameters['sim']),
        np.arange(*parameters['mod']),
        np.arange(*parameters['C']),
        np.arange(*parameters['ksi']),
        np.arange(*parameters['epsilon']),
    ):
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
                'ksi': ksi,
                'epsilon': epsilon
            }

    return best_parameters, best_recall, best_precission, best_f1


def plots(stream: npt.NDArray[np.float64],
          best_parameters: dict,
          labels: npt.NDArray[np.bool_],
          stream_name: str) -> None:
    """
        Method is generating plots for stream including model predictions.

        Args:
            stream (npt.NDArray[np.float64]): stream values vector
            best_parameters (dict): dictionary with best model parameters
            labels (npt.NDArray[np.bool_]): labels vector
            stream_name (str): stream name for plot title
    """
    oesnn_ad = OeSNNAD(stream,
                       window_size=best_parameters['Wsize'],
                       num_in_neurons=best_parameters['NIsize'],
                       num_out_neurons=best_parameters['NOsize'],
                       ts_factor=best_parameters['TS'],
                       mod=best_parameters['mod'],
                       c_factor=best_parameters['C'],
                       epsilon=best_parameters['epsilon'],
                       ksi=best_parameters['ksi'],
                       sim=best_parameters['sim'],
                       beta=best_parameters['Beta'])
    result = oesnn_ad.predict()
    fn_list = []
    fp_list = []
    tn_list = []
    tp_list = []
    for index, (value, predict, label) in enumerate(
            zip(stream.tolist(), result.tolist(), labels.tolist())):
        if predict and label:
            tp_list.append((index, value))
        if predict and not label:
            fp_list.append((index, value))
        if not predict and not label:
            tn_list.append((index, value))
        if not predict and label:
            fn_list.append((index, value))
    plt.figure(figsize=(15, 5))
    plt.title(stream_name)
    plt.plot([x[0] for x in fn_list], [x[1]
             for x in fn_list], '+g', linestyle='None')
    plt.plot([x[0] for x in fp_list], [x[1]
             for x in fp_list], 'sb', linestyle='None')
    plt.plot([x[0] for x in tn_list], [x[1]
             for x in tn_list], '.k', linestyle='None')
    plt.plot([x[0] for x in tp_list], [x[1]
             for x in tp_list], '2r', linestyle='None')
    plt.grid()
    green_plus = mlines.Line2D([], [], color='green', marker='+', linestyle='None',
                               markersize=10, label='FN')
    blue_square = mlines.Line2D([], [], color='blue', marker='s', linestyle='None',
                                markersize=10, label='FP')
    black_dot = mlines.Line2D([], [], color='black', marker='.', linestyle='None',
                              markersize=10, label='TN')
    red_triangle = mlines.Line2D([], [], color='red', marker='2', linestyle='None',
                                 markersize=10, label='TP')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, handles=[
               green_plus, blue_square, black_dot, red_triangle])
    plt.xlabel("Timestamp")
    plt.ylabel("Input value (X)")
    plt.savefig(f"./plots/{stream_name}.png")
    plt.close()


def main() -> None:
    """
        Entry function for program. Function read hiperparameters from json files, stream data from 
        csv files, run hiperparameter tuning and write best results to csv file for every dataset.
    """
    parameters_nab = read_parameters("parameters_NAB.json")
    parameters_yahoo = read_parameters("parameters_Yahoo.json")
    # with open('results/nab_result.csv', 'w', encoding='UTF8') as file:
    #     writer = csv.writer(file)

    #     writer.writerow(("dataset", "filename", "f1", "recall",
    #                     "precission") + tuple(parameters_nab.keys()))
    #     for path in get_all_files_paths("../data/NAB"):
    #         dataset = path.split("/")[-2]
    #         filename = path.split("/")[-1][:-4]
    #         print(f"dataset={dataset}, filename={filename}")
    #         data, labels = get_data_from_path(path, True)
    #         param, recall, precission, f_1 = parameters_tuning(
    #             data, labels, parameters_nab)

    #         if param is not None:
    #             writer.writerow((dataset, filename, f_1, recall,
    #                             precission) + tuple(param.values()))

    #             plots(data, param, labels, f"{dataset}-{filename}")
    #         else:
    #             writer.writerow((dataset, filename, f_1, recall,
    #                             precission) + (None, ) * 10)
    with open('results/yahoo_result.csv', 'w', encoding='UTF8') as file:
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
                plots(data, param, labels, f"{dataset}-{filename}")
            else:
                writer.writerow((dataset, filename, f_1, recall,
                                precission) + (None, ) * 10)


if __name__ == "__main__":
    main()
