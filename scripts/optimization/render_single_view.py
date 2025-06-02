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
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
import yaml

from profile_env import ProfileEnv
from colmap import convert_colmap_camera
from metrics import get_pyfvvdp_metric


# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


# parameters
num_frames_benchmark = 10
num_frames_recorder = 1
z_near = 0.01
z_far = 100.0


def im2tensor(img):
    output = img / 255.0
    output = torch.tensor(output).float()
    output = output.permute(2, 0, 1).unsqueeze(0)
    return output

def render_single_view(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # load params file
    yaml_data = []
    for params in args.params:
        with open(os.path.join(params), "r") as f:
            yaml_data.append(yaml.safe_load(f))

    colmap_data = joblib.load(os.path.join(args.data_dir, args.split + ".pkl"))    
    temp_data = {}

    for image_path, view_params in colmap_data.items():
        if Path(image_path).name == args.image_name:
            temp_data[image_path] = view_params
            width, height = colmap_data[image_path]["image_size"]
    
    colmap_data = temp_data
    
    width = width * args.image_scale
    height = height * args.image_scale

    config = {}
    config_names = {"pred": args.config_pred, "gt": args.config_gt}
    for model in ["pred", "gt"]:
        config[model] = vkgs_py.Config(config_names[model], "immediate")
        config[model].num_frames_benchmark(num_frames_benchmark)
        config[model].num_frames_recorder(num_frames_recorder)
        config[model].res(width, height)
    
    config["gt"].dynamic_res(False);
    config["pred"].dynamic_res(False);
    
    if args.overdraw:
        config["gt"].debug(True)
        config["pred"].debug(True)
        config["gt"].vis_mode("overdraw")
        config["pred"].vis_mode("overdraw")
        config["gt"].vis_scale(450)
        config["pred"].vis_scale(450)
    
    metric_funcs = {}
    metric_funcs["fv"] = get_pyfvvdp_metric(
        width=width,
        height=height,
        mode="standard_fhd",
        foveated=True,
        heatmap="threshold",
        device=torch.device("cpu"),
    )
    metric_funcs["psnr"] = PeakSignalNoiseRatio(data_range=(0, 1)).to(torch.device("cuda", 0))
    metric_funcs["ssim"] = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(torch.device("cuda", 0))
    metric_funcs["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(torch.device("cuda", 0))
    
    # load ground truth data
    engines = {}
    engines["gt"] = vkgs_py.Engine(config["gt"])
    engines["gt"].load_splats(args.input)
    engines["gt"].start()
    
    for image_name, view_params in colmap_data.items():
        model_transform = vkgs_py.mat4_to_numpy(engines["gt"].get_model_matrix())
        sample_state = convert_colmap_camera(view_params, model_transform, z_near, z_far)["sample_state"]
        sample_state.center = [vkgs_py.vec2(0.5)] * num_frames_recorder

        # ground truth rendering
        sample_params = vkgs_py.SampleParams()
        sample_params.num_frames_benchmark = num_frames_benchmark
        sample_params.num_frames_recorder = num_frames_recorder
        sample_params.lod = [[1.0] * config["gt"].num_levels()] * num_frames_recorder
        sample_params.res = [[1.0] * config["gt"].num_levels()] * num_frames_recorder
        sample_params.lod_params = [[vkgs_py.vec4(1.0)] * config["gt"].num_levels()] * num_frames_recorder

        sample_result_gt = engines["gt"].sample(sample_params, sample_state)
        render_gt = np.array(vkgs_py.get_sample_result_numpy(sample_result_gt)[0])
        time_gt = np.median(sample_result_gt.time).item()

    engines["gt"].end()

    # load render data
    engines["pred"] = vkgs_py.Engine(config["pred"])
    engines["pred"].load_splats(args.input)
    engines["pred"].start()

    render_pred = {}
    time_pred = {}
    for image_name, view_params in colmap_data.items():
        for yaml_data_single in yaml_data:
            for name, params in yaml_data_single.items():
                model_transform = vkgs_py.mat4_to_numpy(engines["gt"].get_model_matrix())
                sample_state = convert_colmap_camera(view_params, model_transform, z_near, z_far)["sample_state"]
                sample_state.center = [vkgs_py.vec2(0.5)] * num_frames_recorder

                # predicted rendering
                sample_params = vkgs_py.SampleParams()
                sample_params.num_frames_benchmark = num_frames_benchmark
                sample_params.num_frames_recorder = num_frames_recorder
                sample_params.lod = [[params["lod"]] * config["pred"].num_levels()] * num_frames_recorder
                sample_params.res = [[params["res"]] * config["pred"].num_levels()] * num_frames_recorder
                lod_params = vkgs_py.vec4(*params["lod_params"])
                sample_params.lod_params = [[lod_params] * config["pred"].num_levels()] * num_frames_recorder

                sample_result_pred = engines["pred"].sample(sample_params, sample_state)
                render_pred[name] = np.array(vkgs_py.get_sample_result_numpy(sample_result_pred)[0])
                time_pred[name] = np.median(sample_result_pred.time).item()

    engines["pred"].end()
    
    # compute metrics
    metrics = {}
    for name, render in render_pred.items():
        t_pred = im2tensor(render_pred[name]).to(torch.device("cuda", 0))
        t_gt = im2tensor(render_gt).to(torch.device("cuda", 0))
        metrics[name] = {
            "psnr": metric_funcs["psnr"](t_pred, t_gt).item(),
            "ssim": metric_funcs["ssim"](t_pred, t_gt).item(),
            "lpips": metric_funcs["lpips"](t_pred, t_gt).item(),
        }

    print("GT:")
    print("\tTime: " + str(time_gt))
    if args.overdraw:
        Image.fromarray(render_gt).save(os.path.join(args.output_dir, "gt_overdraw.png"))
    else:
        Image.fromarray(render_gt).save(os.path.join(args.output_dir, "gt.png"))

    for name, render in render_pred.items():
        print(name + ":")
        for metric, value in metrics[name].items():
            print("\t" + metric + ": " + str(value))
        print("\tTime: " + str(time_pred[name]))
        if args.overdraw:
            Image.fromarray(render_pred[name]).save(os.path.join(args.output_dir, name + "_overdraw.png"))
        else:
            Image.fromarray(render_pred[name]).save(os.path.join(args.output_dir, name + ".png"))
        
    if args.visualize:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=len(render_pred)+1,
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        ax[0, 0].imshow(render_gt)
        ax[0, 0].set_title("GT")
        
        index = 1
        for name, render in render_pred.items():
            ax[0, index].imshow(render)
            ax[0, index].set_title(name)
            index += 1

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_gt", type=str, help="config filename", required=True)
    parser.add_argument("--config_pred", type=str, help="config filename", required=True)
    parser.add_argument("--data_dir",type=str, help="data directory", required=True)
    parser.add_argument("--image_name", type=str, help="image name", required=True)
    parser.add_argument("--image_scale", type=int, help="image scale", default=1)
    parser.add_argument("--input", type=str, help="LOD model filename", required=True)
    parser.add_argument("--output_dir", type=str, help="output directory", required=True)
    parser.add_argument("--overdraw", action="store_true", help="overdraw")
    parser.add_argument("--params", nargs="+", type=str, help="SampleParams file [.yaml]", required=True)
    parser.add_argument("--split", type=str, help="data split", default="val")
    parser.add_argument("--visualize", action="store_true", help="visualize optimization process")
    args = parser.parse_args()
    
    # profile env
    profile_env = ProfileEnv()
    profile_env.setup(mode="highest")

    render_single_view(args)
        
    profile_env.shutdown()

    
