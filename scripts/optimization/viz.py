# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import yaml

from data_loader import load_data

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py

MARKER_X_SIZE = 50

def get_max_index(target):
    return max(enumerate(target), key=lambda x: x[1])[0]


def setup_plot():
    fig = plt.figure()
    fig.set_dpi(300)
    return fig


def get_scale(target):
    return [x*10 for x in target]


def scatter1d(target, lod0):
    best_index = get_max_index(target)
    y = [0]*len(lod0)
    fig = setup_plot()
    ax = fig.add_subplot()
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    pc = ax.scatter(lod0, y, s=get_scale(target), c=target, cmap="viridis")
    i = get_max_index(target)
    ax.scatter([lod0[i]], [y[i]], s=MARKER_X_SIZE, c="red", marker="x")
    ax.set_xlabel("foveal LOD (% of splats)")
    ax.set_yticks([])
    cbar = fig.colorbar(pc, ax=ax, orientation="horizontal")
    cbar.set_label("objective function value")
    return fig, ax


def scatter2d(target, lod0, lod1):
    fig = setup_plot()
    ax = fig.add_subplot()
    pc = ax.scatter(lod0, lod1, s=get_scale(target), c=target, cmap="viridis")
    i = get_max_index(target)
    ax.scatter([lod0[i]], [lod1[i]], s=MARKER_X_SIZE, c="red", marker="x")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("foveal LOD (% of splats)")
    ax.set_ylabel("peripheral LOD (% of splats)")
    ax.set_aspect("equal")
    cbar = fig.colorbar(pc, ax=ax)
    cbar.set_label("objective function value")
    return fig, ax


def scatter3d(target, lod0, lod1, lod2):
    fig = setup_plot()
    ax = fig.add_subplot(projection="3d")
    pc = ax.scatter(lod0, lod1, lod2, s=get_scale(target), c=target, cmap="viridis")
    i = get_max_index(target)
    pc = ax.scatter([lod0[i]], [lod1[i]], [lod2[i]], s=MARKER_X_SIZE, c="red", marker="x")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    ax.set_xlabel("foveal LOD (% of splats)")
    ax.set_ylabel("middle LOD (% of splats)")
    ax.set_zlabel("peripheral LOD (% of splats)")
    cbar = fig.colorbar(pc, ax=ax, orientation="horizontal")
    cbar.set_label("objective function value")
    return fig, ax


def visualize_optimizer(optimizer):
    target = []

    num_lod = len([x for x in optimizer.res[0]["params"].keys() if x.startswith("lod")])
    lod = [[] for _ in range(num_lod)]
    
    for iteration in optimizer.res:
        target.append(iteration["target"])
        
        for i in range(num_lod):
            lod[i].append(iteration["params"]["lod"+str(i)] * 100)

    if num_lod == 1:
        fig, ax = scatter1d(target, lod[0])
    elif num_lod == 2:
        fig, ax = scatter2d(target, lod[0], lod[1])
    elif num_lod == 3:
        fig, ax = scatter3d(target, lod[0], lod[1], lod[2])

    ax.set_title("Samples for LOD sampling")
    plt.show()
    

def compute_heatmap_from_unstructured_samples(
    samples: np.ndarray,
    shape: Tuple[int],
):
    assert(samples.shape[1] == 3)
    x, y = np.meshgrid(np.linspace(0, 1, shape[0]),
                       np.linspace(0, 1, shape[1]))
    interpolator = LinearNDInterpolator(list(zip(samples[:, 0], samples[:, 1])), samples[:, 2], fill_value=0)
    heatmap = interpolator(x, y)
    return heatmap


def visualize_gaze_heatmap(
    gt_image: np.ndarray,
    centers: np.ndarray,
    sample_params: List[vkgs_py.SampleParams],
):
    lod_0 = []
    lod_1 = []
    for i in range(len(sample_params)):
        lod_0.append(sample_params[i].lod[0][0])
        lod_1.append(sample_params[i].lod[0][1])
    lod_0 = np.array(lod_0)[:, None]
    lod_1 = np.array(lod_1)[:, None]
    
    lod_0_field = compute_heatmap_from_unstructured_samples(
        np.concatenate((centers, lod_0), axis=-1),
        (gt_image.shape[2], gt_image.shape[1]),
    )  # [H, W]

    lod_1_field = compute_heatmap_from_unstructured_samples(
        np.concatenate((centers, lod_1), axis=-1),
        (gt_image.shape[2], gt_image.shape[1]),
    )  # [H, W]

    max_value = max(np.max(lod_0_field), np.max(lod_1_field))
    lod_0_field = lod_0_field / max_value
    lod_1_field = lod_1_field / max_value

    lod_0_field = cv2.applyColorMap((lod_0_field * 255).astype(np.uint8), cv2.COLORMAP_JET)
    lod_1_field = cv2.applyColorMap((lod_1_field * 255).astype(np.uint8), cv2.COLORMAP_JET)

    alpha = 0.5
    
    fig, axes = plt.subplots(1, 2)
    image_0 = ((lod_0_field * alpha) + (gt_image[0] * (1.0-alpha))).astype(np.uint8)
    image_1 = ((lod_1_field * alpha) + (gt_image[0] * (1.0-alpha))).astype(np.uint8)
    axes[0].imshow(image_0)
    axes[1].imshow(image_1)
    axes[0].set_title("Foveal LOD")
    axes[1].set_title("Peripheral LOD")
    fig.suptitle("LOD levels by region and gaze direction")
    plt.show()


class GazeHeatmapVisualizer:
    def __init__(
        self,
        num_gaze_dir,
    ):
        self.num_gaze_dir = num_gaze_dir
        self.centers = []
        self.sample_params = []


    def insert(
        self,
        center,
        sample_params,
    ):
        self.centers.append(center)
        self.sample_params.append(sample_params)


    def visualize(self, image_gt):
        if len(self.centers) == self.num_gaze_dir:
            visualize_gaze_heatmap(image_gt, np.array(self.centers), self.sample_params)
            self.centers.clear()
            self.sample_params.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str, help="data directory", required=True)
    parser.add_argument("--image_dir", type=str, help="image directory")
    args = parser.parse_args()
    
    data = load_data(os.path.join(args.data_dir, "train.yaml"))
    
    heatmap_visualizer = GazeHeatmapVisualizer()
