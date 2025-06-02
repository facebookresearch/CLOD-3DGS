# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

import joblib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import pyfvvdp
import scipy
import seaborn as sns
import torch
from tqdm import tqdm
import yaml

from profile_env import ProfileEnv
from colmap import convert_colmap_camera
from metrics import get_pyfvvdp_metric, convert_eccentricity_to_radius


# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


# parameters
z_near = 0.01
z_far = 100.0


def test_foveation(args):
    with open(os.path.join(args.data_dir, "models", args.model, "train.yaml"), "r") as f:
        yaml_data = yaml.safe_load(f)
    
    num_frames = 1
    
    # load sample image to get image size (assumes all images are the same size)
    width, height = yaml_data[list(yaml_data.keys())[0]]["metadata"]["res"]
    width = width * args.image_scale
    height = height * args.image_scale

    config = {}
    config_names = {"fov": args.config_fov, "gt": args.config_gt}
    for model in ["fov", "gt"]:
        config[model] = vkgs_py.Config(config_names[model], "immediate")
        config[model].num_frames_benchmark(1)
        config[model].num_frames_recorder(1)
        config[model].res(width, height)
    
    config["gt"].dynamic_res(False);
    config["fov"].dynamic_res(False);

    style = "threshold"
    
    metric_funcs = {}
    metric_funcs["fv"] = get_pyfvvdp_metric(
        width=width,
        height=height,
        mode="standard_fhd",
        foveated=True,
        heatmap=style,
        device=torch.device("cpu"),
    )

    os.makedirs(os.path.join(args.data_dir, "viz"), exist_ok=True)
    with open(os.path.join(args.data_dir, "viz", "foveated_info.txt"), "w") as f:
        f.write(metric_funcs["fv"].get_info_string())
    
    # load ground truth data
    engines = {}
    engines["gt"] = vkgs_py.Engine(config["gt"])
    engines["gt"].load_splats(args.input)
    engines["gt"].start()

    # load foveated data
    engines["fov"] = vkgs_py.Engine(config["fov"])
    engines["fov"].load_splats(args.input)
    engines["fov"].start()

    renders = {}
    renders["gt"] = {}

    progress = tqdm(total=len(yaml_data))
    for image_name, view_params in yaml_data.items():
        sample_state = vkgs_py.SampleState()
        sample_state.pos = [vkgs_py.vec3(*view_params["sample_state"]["pos"][0])]
        quat = view_params["sample_state"]["quat"][0]
        sample_state.quat = [vkgs_py.quat(quat[0], quat[1], quat[2], quat[3])]
        sample_state.center = [vkgs_py.vec2(0.5)] * num_frames
        sample_state.gaze_dir = [vkgs_py.vec3(*view_params["sample_state"]["gaze_dir"][0])]
        sample_state.view_angles = [vkgs_py.ViewFrustumAngles(*view_params["sample_state"]["view_angles"][0])]

        fig, ax = plt.subplots(2, 2)

        # ground truth rendering
        sample_params = vkgs_py.SampleParams()
        sample_params.num_frames_benchmark = 1
        sample_params.num_frames_recorder = num_frames
        sample_params.lod = [[1.0] * config["gt"].num_levels()] * num_frames
        sample_params.res = [[1.0] * config["gt"].num_levels()] * num_frames
        sample_params.lod_params = [[vkgs_py.vec4(1.0)] * config["gt"].num_levels()] * num_frames

        sample_result_gt = engines["gt"].sample(sample_params, sample_state)
        data_pred_gt = np.array(vkgs_py.get_sample_result_numpy(sample_result_gt)[0])
        ax[0, 0].set_title("Ground Truth")
        ax[0, 0].imshow(data_pred_gt)

        # foveated rendering
        sample_params = vkgs_py.SampleParams()
        sample_params.num_frames_benchmark = 1
        sample_params.num_frames_recorder = num_frames
        sample_params.lod = view_params["sample_params"]["lod"]
        sample_params.res = [[1.0] * config["fov"].num_levels()] * num_frames
        sample_params.lod_params = [[vkgs_py.vec4(1.0)] * config["fov"].num_levels()] * num_frames

        sample_result_fov = engines["fov"].sample(sample_params, sample_state)
        data_pred_fov = np.array(vkgs_py.get_sample_result_numpy(sample_result_fov)[0])
        ax[0, 1].set_title("Foveated")
        ax[0, 1].imshow(data_pred_fov)

        # get heatmap
        value, stats = metric_funcs["fv"].predict(data_pred_fov, data_pred_gt, dim_order="HWC")
        if style == "raw":
            heatmap_grayscale = stats["heatmap"][0, 0, 0, :, :].detach().cpu().numpy()
            cmap = plt.get_cmap("viridis")
            heatmap = cmap(heatmap_grayscale)
        elif style == "threshold":
            heatmap = stats["heatmap"][0, :, 0, :, :].permute((1, 2, 0)).detach().cpu().numpy()

        heatmap = (heatmap * 255).astype(np.uint8)

        ax[1, 0].set_title("Heatmap (" + str(round(value.item(), 3)) + ")")
        ax[1, 0].imshow(heatmap)
        
        ax[1, 1].set_title("Foveation LOD Levels")
        labels = [str(x) for x in range(len(sample_params.lod[0]))]
        labels[0] = "fovea"
        labels[-1] = "periphery"
        ax[1, 1].bar(
            labels,
            [x for x in sample_params.lod[0]],
        )
        
        os.makedirs(os.path.join(args.data_dir, "renders", "compare"), exist_ok=True)
        plt.savefig(os.path.join(args.data_dir, "renders", "compare", str(image_name) + ".png"), dpi=300)
        plt.clf()

        progress.update(1)

    progress.close()

    engines["gt"].end()
    engines["fov"].end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_gt", type=str, help="config filename", required=True)
    parser.add_argument("--config_fov", type=str, help="config filename", required=True)
    parser.add_argument("--data_dir",type=str, help="data directory", required=True)
    parser.add_argument("--display", action="store_true", help="display")
    parser.add_argument("--image_scale", type=int, help="image scale", default=1)
    parser.add_argument("--input", type=str, help="LOD model filename", required=True)
    parser.add_argument("--lod_levels", type=float, nargs="*", help="LOD levels to test", default=[1.0])
    parser.add_argument("--mode", type=str, help="comparison mode [lod_ref, lod_levels]", default="lod_ref")
    parser.add_argument("--model", type=str, help="model name", required=True)
    parser.add_argument("--save_images", action="store_true", help="save images")
    parser.add_argument("--skip_grid_search", action="store_true", help="skip grid search")
    parser.add_argument("--visualize", action="store_true", help="visualize optimization process")
    args = parser.parse_args()
    
    if not args.skip_grid_search:
        # profile env
        profile_env = ProfileEnv()
        profile_env.setup(mode="highest")

        test_foveation(args)
        
        profile_env.shutdown()


    
