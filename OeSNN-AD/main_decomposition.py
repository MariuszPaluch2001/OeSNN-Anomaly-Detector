import csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time

from decomposition import DecomposeOeSNNAD
from utils import get_data_from_path, perf_measure, get_all_files_paths
import pandas as pd
import PyEMD

if __name__ == "__main__":

    df_yahoo = pd.read_csv("results/yahoo_result.csv")
    df_nab = pd.read_csv("results/nab_result.csv")

    with open("results/ceemdan_nab_result-committee.csv", "w", encoding='UTF8') as file:
        writer = csv.writer(file)

        writer.writerow(("dataset", "filename", "f1", "recall",
                        "precission", "common_ratio", "time [s]"))
        for path in get_all_files_paths("../data/NAB"):
            dataset = path.split("/")[-2]
            filename = path.split("/")[-1][:-4]

            print(f"Stream: {dataset}/{filename}")

            stream, labels = get_data_from_path(path, True)

            df_stream = df_nab[(df_nab["dataset"] == dataset) & (
                df_nab["filename"] == filename)]
            df_stream.reset_index(drop=True, inplace=True)
            df_stream = df_stream.loc[0]
            if df_stream['f1'] == 0.0:
                writer.writerow((dataset, filename, 0.0, 0.0,
                                0.0, 0.0, 0.0))
                continue

            best_res, best_f1, best_recall, best_precission = None, 0.0, 0.0, 0.0
            t = time.time()
            decomp = DecomposeOeSNNAD(stream, trials=100)
            PyEMD.CEEMDAN.noise_seed(decomp.ceemdan, 1000)

            res = decomp.predict(cemmndan_channels=-1,
                                 merge_type="committee",
                                 window_size=df_stream["Wsize"].astype(
                                     int),
                                 num_out_neurons=df_stream["NOsize"].astype(
                                     int),
                                 num_in_neurons=df_stream["NIsize"].astype(
                                     int),
                                 beta=df_stream["Beta"].astype(
                                     float),
                                 ts_factor=df_stream["TS"].astype(
                                     float),
                                 sim=df_stream["sim"].astype(
                                     float),
                                 mod=df_stream["mod"].astype(
                                     float),
                                 c_factor=df_stream["C"].astype(
                                     float),
                                 ksi=df_stream["ksi"].astype(
                                     float),
                                 epsilon=df_stream["epsilon"].astype(int))
            recall, precission, f1 = perf_measure(res, labels)
            if f1 > best_f1:
                best_res = best_res
                best_f1 = f1
                best_precission = precission
                best_recall = recall

            time_res = time.time() - t
            print(f"Time: {time_res} sec")
            print(
                f"F1 {best_f1}, Recall {best_recall}, Precission {best_precission}")

            writer.writerow((dataset, filename, best_f1, best_recall,
                             best_precission, time_res))
            fn_list = []
            fp_list = []
            tn_list = []
            tp_list = []
            for index, (value, predict, label) in enumerate(
                    zip(stream.tolist(), res.tolist(), labels.tolist())):
                if predict and label:
                    tp_list.append((index, value))
                if predict and not label:
                    fp_list.append((index, value))
                if not predict and not label:
                    tn_list.append((index, value))
                if not predict and label:
                    fn_list.append((index, value))
            plt.figure(figsize=(15, 5))
            plt.title(filename)
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
            plt.savefig(f"plots-decomposition/NAB-{filename}-committee.png")
            plt.close()

    with open("results/ceemdan_yahoo_result-committee.csv", "w", encoding='UTF8') as file:
        writer = csv.writer(file)

        writer.writerow(("dataset", "filename", "f1", "recall",
                        "precission", "common_ratio", "time [s]"))
        for path in get_all_files_paths("../data/Yahoo"):
            dataset = path.split("/")[-2]
            filename = path.split("/")[-1][:-4]

            print(f"Stream: {dataset}/{filename}")

            stream, labels = get_data_from_path(path, False)

            df_stream = df_yahoo[(df_yahoo["dataset"] == dataset) & (
                df_yahoo["filename"] == filename)]
            df_stream.reset_index(drop=True, inplace=True)
            df_stream = df_stream.loc[0]
            if df_stream['f1'] == 0.0:
                writer.writerow((dataset, filename, 0.0, 0.0,
                                0.0, 0.0, 0.0))
                continue

            best_res, best_f1, best_recall, best_precission = None, 0.0, 0.0, 0.0
            t = time.time()
            decomp = DecomposeOeSNNAD(stream, trials=100)
            PyEMD.CEEMDAN.noise_seed(decomp.ceemdan, 1000)

            res = decomp.predict(cemmndan_channels=-1,
                                 merge_type="committee",
                                 window_size=df_stream["Wsize"].astype(
                                     int),
                                 num_out_neurons=df_stream["NOsize"].astype(
                                     int),
                                 num_in_neurons=df_stream["NIsize"].astype(
                                     int),
                                 beta=df_stream["Beta"].astype(
                                     float),
                                 ts_factor=df_stream["TS"].astype(
                                     float),
                                 sim=df_stream["sim"].astype(
                                     float),
                                 mod=df_stream["mod"].astype(
                                     float),
                                 c_factor=df_stream["C"].astype(
                                     float),
                                 ksi=df_stream["ksi"].astype(
                                     float),
                                 epsilon=df_stream["epsilon"].astype(int))
            recall, precission, f1 = perf_measure(res, labels)
            if f1 > best_f1:
                best_res = best_res
                best_f1 = f1
                best_precission = precission
                best_recall = recall

            time_res = time.time() - t
            print(f"Time: {time_res} sec")
            print(
                f"F1 {best_f1}, Recall {best_recall}, Precission {best_precission}")

            writer.writerow((dataset, filename, best_f1, best_recall,
                             best_precission, time_res))
            fn_list = []
            fp_list = []
            tn_list = []
            tp_list = []
            for index, (value, predict, label) in enumerate(
                    zip(stream.tolist(), res.tolist(), labels.tolist())):
                if predict and label:
                    tp_list.append((index, value))
                if predict and not label:
                    fp_list.append((index, value))
                if not predict and not label:
                    tn_list.append((index, value))
                if not predict and label:
                    fn_list.append((index, value))
            plt.figure(figsize=(15, 5))
            plt.title(filename)
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
            plt.savefig(f"plots-decomposition/Yahoo-{filename}-committee.png")
            plt.close()

    with open("results/ceemdan_nab_result-geometric.csv", "w", encoding='UTF8') as file:
        writer = csv.writer(file)

        writer.writerow(("dataset", "filename", "f1", "recall",
                        "precission", "common_ratio", "time [s]"))
        for path in get_all_files_paths("../data/NAB"):
            dataset = path.split("/")[-2]
            filename = path.split("/")[-1][:-4]

            print(f"Stream: {dataset}/{filename}")

            stream, labels = get_data_from_path(path, True)

            df_stream = df_nab[(df_nab["dataset"] == dataset) & (
                df_nab["filename"] == filename)]
            df_stream.reset_index(drop=True, inplace=True)
            df_stream = df_stream.loc[0]
            if df_stream['f1'] == 0.0:
                writer.writerow((dataset, filename, 0.0, 0.0,
                                0.0, 0.0, 0.0))
                continue

            best_res, best_f1, best_recall, best_precission, best_ratio = None, 0.0, 0.0, 0.0, None
            t = time.time()
            for ratio in (0.5, 0.6, 0.7, 0.9, 1.0):
                decomp = DecomposeOeSNNAD(stream, trials=100)
                PyEMD.CEEMDAN.noise_seed(decomp.ceemdan, 1000)

                res = decomp.predict(cemmndan_channels=-1,
                                     merge_type="geometric",
                                     common_ratio=ratio,
                                     window_size=df_stream["Wsize"].astype(
                                         int),
                                     num_out_neurons=df_stream["NOsize"].astype(
                                         int),
                                     num_in_neurons=df_stream["NIsize"].astype(
                                         int),
                                     beta=df_stream["Beta"].astype(
                                         float),
                                     ts_factor=df_stream["TS"].astype(
                                         float),
                                     sim=df_stream["sim"].astype(
                                         float),
                                     mod=df_stream["mod"].astype(
                                         float),
                                     c_factor=df_stream["C"].astype(
                                         float),
                                     ksi=df_stream["ksi"].astype(
                                         float),
                                     epsilon=df_stream["epsilon"].astype(int))
                recall, precission, f1 = perf_measure(res, labels)
                if f1 > best_f1:
                    best_res = best_res
                    best_f1 = f1
                    best_precission = precission
                    best_recall = recall
                    best_ratio = ratio

            time_res = time.time() - t
            print(f"Time: {time_res} sec")
            print(
                f"F1 {best_f1}, Recall {best_recall}, Precission {best_precission} Best ratio {best_ratio}")

            writer.writerow((dataset, filename, best_f1, best_recall,
                             best_precission, best_ratio, time_res))
            fn_list = []
            fp_list = []
            tn_list = []
            tp_list = []
            for index, (value, predict, label) in enumerate(
                    zip(stream.tolist(), res.tolist(), labels.tolist())):
                if predict and label:
                    tp_list.append((index, value))
                if predict and not label:
                    fp_list.append((index, value))
                if not predict and not label:
                    tn_list.append((index, value))
                if not predict and label:
                    fn_list.append((index, value))
            plt.figure(figsize=(15, 5))
            plt.title(filename)
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
            plt.savefig(f"plots-decomposition/NAB-{filename}-geometric.png")
            plt.close()

    with open("results/ceemdan_yahoo_result-geometric.csv", "w", encoding='UTF8') as file:
        writer = csv.writer(file)

        writer.writerow(("dataset", "filename", "f1", "recall",
                        "precission", "common_ratio", "time [s]"))
        for path in get_all_files_paths("../data/Yahoo"):
            dataset = path.split("/")[-2]
            filename = path.split("/")[-1][:-4]

            print(f"Stream: {dataset}/{filename}")

            stream, labels = get_data_from_path(path, False)

            df_stream = df_yahoo[(df_yahoo["dataset"] == dataset) & (
                df_yahoo["filename"] == filename)]
            df_stream.reset_index(drop=True, inplace=True)
            df_stream = df_stream.loc[0]
            if df_stream['f1'] == 0.0:
                writer.writerow((dataset, filename, 0.0, 0.0,
                                0.0, 0.0, 0.0))
                continue

            best_res, best_f1, best_recall, best_precission, best_ratio = None, 0.0, 0.0, 0.0, None
            t = time.time()
            for ratio in (0.5, 0.6, 0.7, 0.9, 1.0):
                decomp = DecomposeOeSNNAD(stream, trials=100)
                PyEMD.CEEMDAN.noise_seed(decomp.ceemdan, 1000)

                res = decomp.predict(cemmndan_channels=-1,
                                     merge_type="geometric",
                                     common_ratio=ratio,
                                     window_size=df_stream["Wsize"].astype(
                                         int),
                                     num_out_neurons=df_stream["NOsize"].astype(
                                         int),
                                     num_in_neurons=df_stream["NIsize"].astype(
                                         int),
                                     beta=df_stream["Beta"].astype(
                                         float),
                                     ts_factor=df_stream["TS"].astype(
                                         float),
                                     sim=df_stream["sim"].astype(
                                         float),
                                     mod=df_stream["mod"].astype(
                                         float),
                                     c_factor=df_stream["C"].astype(
                                         float),
                                     ksi=df_stream["ksi"].astype(
                                         float),
                                     epsilon=df_stream["epsilon"].astype(int))
                recall, precission, f1 = perf_measure(res, labels)
                if f1 > best_f1:
                    best_res = best_res
                    best_f1 = f1
                    best_precission = precission
                    best_recall = recall
                    best_ratio = ratio

            time_res = time.time() - t
            print(f"Time: {time_res} sec")
            print(
                f"F1 {best_f1}, Recall {best_recall}, Precission {best_precission} Best ratio {best_ratio}")

            writer.writerow((dataset, filename, best_f1, best_recall,
                             best_precission, best_ratio, time_res))
            fn_list = []
            fp_list = []
            tn_list = []
            tp_list = []
            for index, (value, predict, label) in enumerate(
                    zip(stream.tolist(), res.tolist(), labels.tolist())):
                if predict and label:
                    tp_list.append((index, value))
                if predict and not label:
                    fp_list.append((index, value))
                if not predict and not label:
                    tn_list.append((index, value))
                if not predict and label:
                    fn_list.append((index, value))
            plt.figure(figsize=(15, 5))
            plt.title(filename)
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
            plt.savefig(f"plots-decomposition/Yahoo-{filename}-geometric.png")
            plt.close()
