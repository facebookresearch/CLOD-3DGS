# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_dataset(scene):
    datasets = {
        "Mip-NeRF360": ["bicycle", "garden", "stump", "room", "counter", "kitchen", "bonsai", "flowers", "treehill"],
        "Tanks&Temples": ["truck", "train"],
        "Deep Blending": ["drjohnson", "playroom"],
    }
    for dataset_name, dataset_scenes in datasets.items():
        if scene in dataset_scenes:
            return dataset_name


def plot_dist_lod_levels(result_dir):
    # metrics [metric, scene, exp_name]
    metrics = {}
    
    # global variables
    exp_names = []
    scenes = []
    datasets = ["Deep Blending"]

    # populate metrics
    scenes = os.listdir(os.path.join(result_dir))
    scenes = [x for x in scenes if os.path.isdir(os.path.join(result_dir, x))]
    for scene in scenes:
        with open(os.path.join(result_dir, scene, "metrics.csv"), "r") as f:
            headers = f.readline()
            exp_names = [x for x in headers.strip().split(",")[1:]]

            for line in f:
                row = line.strip().split(",")
                row_metric = row[0]
                row_lod = row[1:]
                    
                # create entries if empty
                if row_metric not in metrics:
                    metrics[row_metric] = {}
                if scene not in metrics[row_metric]:
                    metrics[row_metric][scene] = {}

                for lod_i in range(len(exp_names)):
                    exp_name = exp_names[lod_i]
                    metrics[row_metric][scene][exp_name] = float(row_lod[lod_i])

    # compile data for each dataset
    metrics_combined = {}
    for metric in metrics.keys():
        metrics_combined[metric] = {}
        for scene in scenes:
            dataset = get_dataset(scene)
                
            if dataset not in metrics_combined[metric]:
                metrics_combined[metric][dataset] = {}
                
            for exp_name in metrics[metric][scene].keys():
                if exp_name not in metrics_combined[metric][dataset]:
                    metrics_combined[metric][dataset][exp_name] = []
                    
                value = metrics[metric][scene][exp_name]
                metrics_combined[metric][dataset][exp_name].append(value)


    # compute mean
    for metric in metrics_combined.keys():
        for scene in metrics_combined[metric].keys():
            for exp_name in metrics_combined[metric][scene].keys():
                metrics_combined[metric][scene][exp_name] = np.mean(metrics_combined[metric][scene][exp_name])

    # export .csv
    csv_metrics = ["PSNR", "SSIM", "LPIPS", "Time"]
    header_dataset = ["Dataset"]
    header_metric = ["Metric"]
            
    for dataset in datasets:
        for metric in csv_metrics:
            header_dataset.append(dataset)
            header_metric.append(str(metric))
            
    data = {}
    data_csv = {}
    for metric in csv_metrics:
        if metric not in data:
            if metric == "Time":
                data["Time (ms)"] = {}
                data["Time (x)"] = {}
            else:
                data[metric] = {}

        for exp_name in exp_names:    
            if exp_name not in ["gt"]:
                lod_distance = float(exp_name.split("___")[0].replace("_", "."))
                lod_percentage = int(float(exp_name.split("___")[1].replace("_", ".")) * 100)
                
                if metric == "Time":
                    if lod_distance not in data["Time (ms)"]:
                        data["Time (ms)"][lod_distance] = {}
                        data["Time (x)"][lod_distance] = {}
                else:
                    if lod_distance not in data[metric]:
                        data[metric][lod_distance] = {}

                if lod_percentage == 100:
                    continue

                if metric == "Time":
                    data["Time (ms)"][lod_distance][lod_percentage] = metrics_combined[metric][dataset][exp_name]
                    data["Time (x)"][lod_distance][lod_percentage] = metrics_combined[metric][dataset]["gt"] / metrics_combined[metric][dataset][exp_name]
                else:
                    data[metric][lod_distance][lod_percentage] = metrics_combined[metric][dataset][exp_name]

    for dataset in metrics_combined[metric].keys():
        for metric in csv_metrics:
            for exp_name in exp_names:
                if exp_name not in data_csv:
                    data_csv[exp_name] = [exp_name]
                
                value = metrics_combined[metric][dataset][exp_name]
                data_csv[exp_name].append("{:.3f}".format(value))
            
    with open(os.path.join(result_dir, "combined_metrics.csv"), "w") as f:
        f.writelines(",".join(header_dataset) + "\n")
        f.writelines(",".join(header_metric) + "\n")
        for exp_name in exp_names:
            f.writelines(",".join(data_csv[exp_name]) + "\n")

    # helps with copy/pasting into LaTeX document
    with open(os.path.join(result_dir, "combined_metrics_latex.txt"), "w") as f:
        f.writelines(" & ".join(header_dataset) + " \\\\\n")
        f.writelines(" & ".join(header_metric) + " \\\\\n")
        for exp_name in exp_names:
            f.writelines(" & ".join(data_csv[exp_name]) + " \\\\\n")

    # create plots
    dataset = "Deep Blending"
    
    # plot LPIPS
    plt.clf()
    data_frame = pd.DataFrame.from_dict(data["LPIPS"], orient="columns")
    sns.heatmap(data_frame, annot=True, square=True, cbar_kws={"shrink": 0.5}, cmap="viridis")
    plt.xlabel("max distance (m)")
    plt.ylabel("min LOD (% splats)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "dist_lod_lpips.pdf"))

    # plot time
    plt.clf()
    data_frame_ms = pd.DataFrame.from_dict(data["Time (ms)"], orient="columns")
    data_frame_x = pd.DataFrame.from_dict(data["Time (x)"], orient="columns")
    data_frame_x = data_frame_x.astype(str)
    data_frame_x = data_frame_x.applymap(lambda x: f"({float(x):.2f}x)")
    sns.heatmap(data_frame_ms, annot=False, square=True, cbar_kws={"shrink": 0.5}, cmap="viridis")
    sns.heatmap(data_frame_ms, annot=data_frame_x, square=True, annot_kws={"va": "top"}, cmap="viridis", cbar=False, fmt="s")
    sns.heatmap(data_frame_ms, annot=data_frame_ms, square=True, annot_kws={"va": "bottom"}, cmap="viridis", cbar=False, fmt="0.3f")
    plt.xlabel("max distance (m)")
    plt.ylabel("min LOD (% splats)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "dist_lod_time.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, help="results directory", required=True)
    args = parser.parse_args()
    
    plot_dist_lod_levels(args.result_dir)
