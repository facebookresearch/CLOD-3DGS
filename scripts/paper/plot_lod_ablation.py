# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def get_dataset(scene):
    datasets = {
        "Mip-NeRF360": ["bicycle", "garden", "stump", "room", "counter", "kitchen", "bonsai", "flowers", "treehill"],
        "Tanks&Temples": ["truck", "train"],
        "Deep Blending": ["drjohnson", "playroom"],
    }
    for dataset_name, dataset_scenes in datasets.items():
        if scene in dataset_scenes:
            return dataset_name


def plot_lod_levels(result_dir):
    methods = os.listdir(result_dir)
    methods = [x for x in methods if os.path.isdir(os.path.join(result_dir, x))]
    methods = [x for x in methods if x != "images"]

    # metrics [metric, method, scene, lod_level]
    metrics = {}
    
    # global variables
    lod_levels = []
    scenes = []
    datasets = ["Mip-NeRF360", "Tanks&Temples", "Deep Blending"]

    # populate metrics
    for method in methods:
        scenes = os.listdir(os.path.join(result_dir, method))
        for scene in scenes:
            with open(os.path.join(result_dir, method, scene, "metrics.csv"), "r") as f:
                headers = f.readline()
                lod_levels = [float(x) for x in headers.strip().split(",")[1:]]

                for line in f:
                    row = line.strip().split(",")
                    row_metric = row[0]
                    row_lod = row[1:]
                    
                    # create entries if empty
                    if row_metric not in metrics:
                        metrics[row_metric] = {}
                    if method not in metrics[row_metric]:
                        metrics[row_metric][method] = {}
                    if scene not in metrics[row_metric][method]:
                        metrics[row_metric][method][scene] = {}

                    for lod_i in range(len(lod_levels)):
                        lod_level = lod_levels[lod_i]
                        metrics[row_metric][method][scene][lod_level] = float(row_lod[lod_i])

    # compile data for each dataset
    metrics_combined = {}
    for metric in metrics.keys():
        metrics_combined[metric] = {}
        for method in methods:
            metrics_combined[metric][method] = {}
            for scene in scenes:
                dataset = get_dataset(scene)
                
                if dataset not in metrics_combined[metric][method]:
                    metrics_combined[metric][method][dataset] = {}
                
                for lod_level in metrics[metric][method][scene].keys():
                    if lod_level not in metrics_combined[metric][method][dataset]:
                        metrics_combined[metric][method][dataset][lod_level] = []
                    
                    value = metrics[metric][method][scene][lod_level]
                    metrics_combined[metric][method][dataset][lod_level].append(value)


    # compute mean
    for metric in metrics_combined.keys():
        for method in metrics_combined[metric].keys():
            for scene in metrics_combined[metric][method].keys():
                for lod_level in metrics_combined[metric][method][scene].keys():
                    metrics_combined[metric][method][scene][lod_level] = np.mean(metrics_combined[metric][method][scene][lod_level])

    # export .csv
    csv_metrics = ["PSNR", "SSIM", "LPIPS", "Time"]
    for method in methods:
        header_dataset = ["Dataset"]
        header_metric = ["Metric"]
            
        for dataset in datasets:
            for metric in csv_metrics:
                header_dataset.append(dataset)
                header_metric.append(str(metric))
            
        data = {}
        for lod_level in lod_levels:
            data[lod_level] = [str(lod_level)]
            
        for dataset in metrics_combined[metric][method].keys():
            for metric in csv_metrics:
                for lod_level in lod_levels:
                    value = metrics_combined[metric][method][dataset][lod_level]
                    data[lod_level].append("{:.2f}".format(value))
            
        with open(os.path.join(result_dir, method + "_combined_metrics.csv"), "w") as f:
            f.writelines(",".join(header_dataset) + "\n")
            f.writelines(",".join(header_metric) + "\n")
            for lod_level in lod_levels:
                f.writelines(",".join(data[lod_level]) + "\n")

        # helps with copy/pasting into LaTeX document
        with open(os.path.join(result_dir, method + "_combined_metrics_latex.txt"), "w") as f:
            f.writelines(" & ".join(header_dataset) + " \\\\\n")
            f.writelines(" & ".join(header_metric) + " \\\\\n")
            for lod_level in lod_levels:
                f.writelines(" & ".join(data[lod_level]) + " \\\\\n")

    # create plots
    dataset = "Mip-NeRF360"
    plot_metrics = [x for x in list(metrics_combined.keys()) if x != "Time"]
    method_label = {
        "3dgs_mcmc": "3DGS-MCMC",
        "clod": "CLOD (Ours)",
    }
    
    for metric in plot_metrics:
        plt.rcParams.update({"font.size": 16})
        
        plt.clf()
        plt.xlabel("Time (ms)")
        plt.ylabel(metric)
        for method in methods:
            time = []
            qual = []
            for lod_level in lod_levels:
                time.append(metrics_combined["Time"][method][dataset][lod_level])
                qual.append(metrics_combined[metric][method][dataset][lod_level])
            plt.plot(time, qual, "o-", label=method_label[method])
            for lod_i in range(len(lod_levels)):
                plt.annotate(str(round(lod_levels[lod_i] * 100)) + "%", (time[lod_i], qual[lod_i]), xytext=(2, 2), textcoords="offset points")
        plt.legend(loc="best")
        plt.savefig(os.path.join(result_dir, metric + ".pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, help="results directory", required=True)
    args = parser.parse_args()
    
    plot_lod_levels(args.result_dir)
