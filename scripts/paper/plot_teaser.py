# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Produces teaser image
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw


def plot_teaser(image_name, result_dir, scene, aspect, offset):
    methods = ["clod", "3dgs_mcmc"]
    
    test_image = Image.open(os.path.join(result_dir, methods[0], scene, "gt", image_name))
    width = test_image.size[0]
    height = test_image.size[1]
    
    if aspect:
        c_y = (height // 2) + offset[1]
        height = int(width / aspect)
    
    # plot images
    num_pixels_x = 2
    num_pixels_y = 4
    canvas = np.zeros_like(np.array(test_image).astype(np.float32))
    canvas = canvas[c_y-height//2:c_y+height//2]
    for method_i in range(len(methods)):
        method = methods[method_i]
        lod_levels = os.listdir(os.path.join(os.path.join(result_dir, method, scene)))
        lod_levels = [x for x in lod_levels if os.path.isdir(os.path.join(os.path.join(result_dir, method, scene, x)))]
        lod_levels = [x for x in lod_levels if x != "gt"]
    
        for lod_i in range(len(lod_levels)):
            lod_level = lod_levels[lod_i]
            image = np.array(Image.open(os.path.join(result_dir, method, scene, lod_level, image_name))).astype(np.float32)
            image = image[c_y-height//2:c_y+height//2]

            w = width // len(lod_levels)
            h = height // len(methods)

            min_x = w * (lod_i)
            max_x = w * (lod_i + 1)
            min_y = h * (method_i)
            max_y = h * (method_i + 1)

            if lod_i < len(lod_levels) - 1:
                max_x = max_x - num_pixels_x
            if lod_i > 0:
                min_x = min_x + num_pixels_x
            if method_i < len(methods) - 1:
                max_y = max_y - num_pixels_y
            if method_i > 0:
                min_y = min_y + num_pixels_y

            canvas[min_y:max_y, min_x:max_x] = image[min_y:max_y, min_x:max_x]
        
        index = 0
        for lod_i in range(len(lod_levels)-1):
            lod_level = lod_levels[lod_i]
                
            index += 1
    
    centers_x = np.linspace(0, 1.0, len(lod_levels) + 1)
    centers_y = np.linspace(0, 1.0, len(methods) + 1)
    center_offset_x = (centers_x[1] - centers_x[0]) / 2
    center_offset_y = (centers_y[1] - centers_y[0]) / 2

    height, width, _ = canvas.shape

    # export image
    fig = plt.figure(frameon=False)
    fig_width = (width / width) * 8.5
    fig_height = (height / width) * 8.5
    fig.set_size_inches(fig_width, fig_height)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])    
    fig.add_axes(ax)

    # add text
    index = 0
    font_path = os.path.join(os.getenv("LOCALAPPDATA"), "Microsoft/Windows/Fonts/Roboto-Regular.ttf")
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    for lod_level in lod_levels:
        lod_amount = round(float(lod_level.split("lod_")[1].replace("_", ".")) * 100)
        ax.text(
            (centers_x[index] + center_offset_x) * width,
            0.035 * height,
            str(lod_amount) + "%",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=15 * (height / 500),
            fontproperties=prop,
            color="white",
        )
        index += 1

    # add text background
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            width,
            height * 0.07,
            linewidth=0,
            facecolor="black",
            alpha=0.7,
            fill=True,
        )
    )

    # add text background
    method_names = {
        "clod": "Ours",
        "3dgs_mcmc": "3DGS-MCMC",
    }
    
    index = 0
    for method in methods:
        ax.text(
            0.0175 * width,
            (centers_y[index] + center_offset_y) * height,
            method_names[method],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=15 * (height / 500),
            fontproperties=prop,
            color="white",
            rotation=90,
        )
        index += 1

    ax.add_patch(
        patches.Rectangle(
            (0, height * 0.07),
            width * 0.035,
            height,
            linewidth=0,
            facecolor="black",
            alpha=0.7,
            fill=True,
        )
    )

    ax.imshow(Image.fromarray(canvas.astype(np.uint8)))
    plt.savefig(os.path.join(result_dir, "teaser.pdf"), dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", type=float, help="aspect ratio", default=None)
    parser.add_argument("--image_name", type=str, help="scene", required=True)
    parser.add_argument("--result_dir", type=str, help="results directory", required=True)
    parser.add_argument("--scene", type=str, help="scene", required=True)
    parser.add_argument("--offset", nargs=2, type=int, help="pixel offset")
    args = parser.parse_args()
    
    plot_teaser(args.image_name, args.result_dir, args.scene, args.aspect, args.offset)
