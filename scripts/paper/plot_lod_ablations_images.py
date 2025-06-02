# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Produces grid of images for LOD comparison with 3DGS-MCMC
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from PIL import Image


def plot_lod_levels_images(image_name, result_dir, scene, square_mode=False):
    methods = os.listdir(result_dir)
    methods = [x for x in methods if os.path.isdir(os.path.join(result_dir, x))]
    methods = [x for x in methods if x != "images"]

    lod_levels = os.listdir(os.path.join(os.path.join(result_dir, methods[0], scene)))
    lod_levels = [x for x in lod_levels if os.path.isdir(os.path.join(os.path.join(result_dir, methods[0], scene, x)))]
    
    images = []
    for _ in methods:
        images.append([])
        for _ in lod_levels:
            images[-1].append(None)
    
    # plot images
    fig, ax = plt.subplots(len(methods), len(lod_levels), sharex=True, sharey=True)
    for method_i in range(len(methods)):
        for lod_i in range(len(lod_levels)):
            method = methods[method_i]
            lod_level = lod_levels[lod_i]
            images[method_i][lod_i] = Image.open(os.path.join(result_dir, method, scene, lod_level, image_name))
            ax[method_i, lod_i].imshow(images[method_i][lod_i])

    # export images
    image_dir = os.path.join(result_dir, "images", scene, image_name.split(".")[0])
    os.makedirs(image_dir, exist_ok=True)
    def save_images(event):
        min_x = math.ceil(ax[0, 0].get_xlim()[0])
        min_y = math.ceil(ax[0, 0].get_ylim()[1])
        max_x = math.ceil(ax[0, 0].get_xlim()[1])
        max_y = math.ceil(ax[0, 0].get_ylim()[0])

        if square_mode:
            center_x = (max_x + min_x) // 2
            center_y = (max_y + min_y) // 2
            width = np.minimum(max_x - min_x, max_y - min_y) // 2
            min_x = center_x - width
            max_x = center_x + width
            min_y = center_y - width
            max_y = center_y + width

        for method_i in range(len(methods)):
            os.makedirs(os.path.join(image_dir, methods[method_i]), exist_ok=True)
            for lod_i in range(len(lod_levels)):
                sub_image = images[method_i][lod_i]
                sub_image = sub_image.crop((min_x, min_y, max_x, max_y))
                sub_image.save(os.path.join(image_dir, methods[method_i], lod_levels[lod_i] + ".png"))
        print("Saved images")

    # create save callback
    ax_button = plt.axes([0.05, 0.05, 0.1, 0.1])
    save_button = Button(ax_button, label="Save", hovercolor="lightgray")
    save_button.on_clicked(save_images)
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, help="scene", required=True)
    parser.add_argument("--result_dir", type=str, help="results directory", required=True)
    parser.add_argument("--scene", type=str, help="scene", required=True)
    args = parser.parse_args()
    
    plot_lod_levels_images(args.image_name, args.result_dir, args.scene, square_mode=True)
