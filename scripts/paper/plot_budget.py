# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_budget(csv_file):
    data = {}
    frame = []
    with open(csv_file, newline="") as csv_file_data:
        csv_reader = csv.reader(csv_file_data, delimiter=",", quotechar="|")
        headers = next(csv_reader)
        for header in headers[1:]:
            data[header] = []

        for row in csv_reader:
            frame.append(int(row[0]))
            index = 1
            for header in headers[1:]:
                data[header].append(float(row[index]))
                index += 1

    plt.clf()
    plt.axline((0, 1.5), (1199, 1.5), linestyle="dashed", color="gray")
    for header in data.keys():
        plt.plot(np.arange(0, len(data[header])), data[header], "-", label=header)
    fig = plt.gcf()
    fig.set_size_inches(6, 2)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Frame")
    plt.ylabel("Time (ms)")
    plt.tight_layout()
    plt.show()
    #plt.savefig(os.path.join("budget.pdf"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, help=".csv file", required=True)
    args = parser.parse_args()
    
    plot_budget(args.csv_file)
