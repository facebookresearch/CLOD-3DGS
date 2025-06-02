# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from gaussians import Gaussians
from geometry import surface_area_ellipsoid
from utils_io import read_ply, save_ply


def sigmoid(x):
    return 1.0 / (1.0 * np.exp(-x))


def plot_opacity(gaussian_params):
    plt.clf()
    x = sigmoid(gaussian_params["opacity"])

    plt.hist(x, bins=20, range=[0, 1])
    plt.title("Opacity frequency")
    plt.xlabel("opacity")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig("opacity.png", dpi=300)


def plot_surface_area(gaussian_params):
    plt.clf()
    x = surface_area_ellipsoid(
        np.exp(gaussian_params["scale_0"]),
        np.exp(gaussian_params["scale_1"]),
        np.exp(gaussian_params["scale_2"]),
    )

    plt.hist(x, bins=20, range=[0, np.percentile(x, 95)])
    plt.title("Surface area frequency")
    plt.xlabel("surface area ($m^2$)")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig("surface_area.png", dpi=300)


def plot_surface_area_vs_opacity(gaussian_params):
    plt.clf()
    sa = surface_area_ellipsoid(
        np.exp(gaussian_params["scale_0"]),
        np.exp(gaussian_params["scale_1"]),
        np.exp(gaussian_params["scale_2"]),
    )
    opacity = sigmoid(gaussian_params["opacity"])
    res = 100
    heatmap = np.zeros((res, res))

    sa_width = (np.max(np.log(sa)) - np.min(np.log(sa))) / res
    opacity_width =(np.max(np.log(opacity)) - np.min(np.log(opacity))) / res

    x = np.log(sa) - np.min(np.log(sa))
    y = np.log(opacity) - np.min(np.log(opacity))

    x = np.clip((x / sa_width).astype(np.int32), 0, res - 1)
    y = np.clip((y / opacity_width).astype(np.int32), 0, res - 1)

    for i in range(res):
        for j in range(res):
            count = (x == i) & (y == j)
            heatmap[i, j] = np.sum(count)

    plt.imshow(heatmap)
    plt.title("Surface area vs. opacity")
    plt.xlabel("surface area [log scale]")
    plt.ylabel("opacity [log scale]")
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.tight_layout()
    plt.savefig("surface_area_vs_opacity.png", dpi=300)


def stats_gaussians(
    src_filename,
):
    print("Reading input Gaussians")
    gaussian_params = read_ply(src_filename)

    print("Plotting graphs")
    plot_opacity(gaussian_params)
    plot_surface_area(gaussian_params)
    plot_surface_area_vs_opacity(gaussian_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source filename", required=True)
    args = parser.parse_args()

    stats_gaussians(
        src_filename=args.src,
    )
