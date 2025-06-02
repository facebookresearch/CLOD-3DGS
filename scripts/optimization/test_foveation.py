# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Performs grid search over different foveation parameters
"""
import argparse
import math
import os
import sys
import time

import joblib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import pyfvvdp
import scipy
import scipy.stats as stats
import seaborn as sns
import torch
from tqdm import tqdm
import yaml

from profile_env import ProfileEnv
from colmap import convert_colmap_camera
from metrics import get_pyfvvdp_metric, convert_eccentricity_to_radius
from view_sampler_foveation import ViewSamplerFoveation


# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


# parameters
num_frames_benchmark = 10
num_frames_recorder = 1
z_near = 0.01
z_far = 100.0


def test_foveation(args, res_levels, ecc_levels):
    colmap_data = joblib.load(os.path.join(args.data_dir, "train.pkl"))
    #colmap_data_key = list(colmap_data.keys())[0]
    #colmap_data = {colmap_data_key: colmap_data[colmap_data_key]}
    
    os.makedirs(os.path.join(args.data_dir, "viz", args.exp_name), exist_ok=True)
    with open(os.path.join(args.data_dir, "viz", args.exp_name, "cmd.txt"), "w") as f:
        cmd_line = " ".join(["python"] + sys.argv)
        f.write(cmd_line)

    num_frames = 1
    
    # load sample image to get image size (assumes all images are the same size)
    width, height = colmap_data[list(colmap_data.keys())[0]]["image_size"]
    width = width * args.image_scale
    height = height * args.image_scale

    radii_levels = [convert_eccentricity_to_radius(x, width, height) for x in ecc_levels]
    
    config = {}
    config_names = {"fov": args.config_fov, "gt": args.config_gt}
    for model in ["fov", "gt"]:
        config[model] = vkgs_py.Config(config_names[model], "immediate")
        config[model].num_frames_benchmark(num_frames_benchmark)
        config[model].num_frames_recorder(num_frames_recorder)
        config[model].res(width, height)
    
    config["gt"].dynamic_res(False);
    config["fov"].dynamic_res(True);

    metric_funcs = {}
    metric_funcs["fv"] = get_pyfvvdp_metric(
        width=width,
        height=height,
        mode="standard_fhd",
        foveated=True,
        device=torch.device("cpu"),
    )

    with open(os.path.join(args.data_dir, "viz", args.exp_name, "foveated_info.txt"), "w") as f:
        f.write(metric_funcs["fv"].get_info_string())
    
    # load ground truth data
    engines = {}
    engines["gt"] = vkgs_py.Engine(config["gt"], False)
    engines["gt"].load_splats(args.input)
    engines["gt"].start()

    # results table
    results = {"time": {}, "FovVideoVDP": {}}

    # get ground truth results
    results["time"]["gt"] = {}
    results["FovVideoVDP"]["gt"] = {}
    results["params"] = {}
    
    sample_params = vkgs_py.SampleParams()
    sample_params.num_frames_benchmark = num_frames_benchmark
    sample_params.num_frames_recorder = num_frames_recorder
    sample_params.lod = [[1.0] * config["gt"].num_levels()] * num_frames_recorder
    sample_params.res = [[1.0] * config["gt"].num_levels()] * num_frames_recorder
    sample_params.lod_params = [[vkgs_py.vec4(1.0)] * config["gt"].num_levels()] * num_frames_recorder

    renders = {}
    renders["gt"] = {}
    
    num_images = len(colmap_data)
    if args.eval_every is not None:
        num_images = math.ceil(num_images / args.eval_every)
    
    # render ground truth
    progress = tqdm(total=num_images)
    index = 0
    for image_path, view_params in colmap_data.items():
        if args.eval_every and index % args.eval_every:
            index += 1
            continue

        model_transform = vkgs_py.mat4_to_numpy(engines["gt"].get_model_matrix())
        sample_state = convert_colmap_camera(view_params, model_transform, z_near, z_far)["sample_state"]
        sample_state.center = [vkgs_py.vec2(0.5)] * num_frames_recorder        

        sample_result = engines["gt"].sample(sample_params, sample_state)
        data_pred = vkgs_py.get_sample_result_numpy(sample_result)
        renders["gt"][image_path] = np.array(data_pred[0])

        results["time"]["gt"][image_path] = np.median(sample_result.time).item()
        results["FovVideoVDP"]["gt"][image_path] = 10.0
        index += 1
        progress.update(1)
    progress.close()

    results["time"]["gt"] = results["time"]["gt"]
    results["FovVideoVDP"]["gt"] = results["FovVideoVDP"]["gt"]

    engines["gt"].end()

    # grid search over foveation parameters
    for i in range(len(radii_levels)):        
        radii_level = radii_levels[i]
        ecc_level = ecc_levels[i]
        renders[ecc_level] = {}
        
        results["time"][ecc_level] = {}
        results["FovVideoVDP"][ecc_level] = {}

        results["params"][ecc_level] = {}
        
        config["fov"].radii_levels([radii_level, 0])

        for res_level in res_levels:
            renders[ecc_level][res_level] = {}
            
            results["time"][ecc_level][res_level] = []
            results["FovVideoVDP"][ecc_level][res_level] = []

            results["params"][ecc_level][res_level] = {"lod": []}
            
            sample_params = vkgs_py.SampleParams()
            sample_params.num_frames_benchmark = num_frames_benchmark
            sample_params.num_frames_recorder = num_frames_recorder
            sample_params.lod = [[1.0] * config["fov"].num_levels()] * num_frames_recorder
            sample_params.res = [[1.0, res_level]] * num_frames_recorder
            sample_params.lod_params = [[vkgs_py.vec4(1.0)] * config["fov"].num_levels()] * num_frames_recorder

            # view sampler
            view_sampler = ViewSamplerFoveation(
                config["fov"],
                None, #engines["fov"],
                metric_funcs["fv"],
                time_budget=args.time,
            )

            print("Eccentricity:", ecc_level, "Res:", res_level)
            progress = tqdm(total=num_images)
            index = 0
            for image_path, view_params in colmap_data.items():
                if args.eval_every and index % args.eval_every:
                    index += 1
                    continue

                engines["fov"] = vkgs_py.Engine(config["fov"], False)
                engines["fov"].load_splats(args.input)
                engines["fov"].start()
                view_sampler.engine = engines["fov"]

                model_transform = vkgs_py.mat4_to_numpy(engines["fov"].get_model_matrix())
                sample_state = convert_colmap_camera(view_params, model_transform, z_near, z_far)["sample_state"]
                sample_state.center = [vkgs_py.vec2(0.5)] * num_frames_recorder

                gt_image = renders["gt"][image_path]
                gt_time = results["time"]["gt"][image_path]
                view_sampler_results = view_sampler.train(sample_state, res_level, gt_image, gt_time)
                sample_params = view_sampler_results["params"]
                
                sample_result = engines["fov"].sample(sample_params, sample_state)
                data_pred = vkgs_py.get_sample_result_numpy(sample_result)
                render = np.array(data_pred[0])

                if args.save_images:
                    renders[ecc_level][res_level][image_path] = render

                value, _ = metric_funcs["fv"].predict(render, renders["gt"][image_path], dim_order="HWC")
                results["time"][ecc_level][res_level].append(np.median(sample_result.time).item())
                results["FovVideoVDP"][ecc_level][res_level].append(torch.median(value).item())
                results["params"][ecc_level][res_level]["lod"].append(sample_params.lod)

                index += 1

                progress.update(1)
                engines["fov"].end()

            progress.close()
        

    if args.save_images:
        font = font_manager.FontProperties(family="sans\-serif")
        font = ImageFont.truetype(font_manager.findfont(font), 32)
        for image_path, view_params in colmap_data.items():
            image_name = Path(image_path).stem
            dir_name = os.path.join(args.data_dir, "viz", args.time, image_name)
            os.makedirs(dir_name, exist_ok=True)
            
            # save non-foveated
            image = Image.fromarray(renders["gt"][image_path])
            draw = ImageDraw.Draw(image, "RGBA")
            draw.rectangle((0, 0, 200, 100), fill=(0, 0, 0, 180))
            draw.text((10, 10), "Res: 1.0", (255, 255, 255), font=font)
            draw.text((10, 50), "Ecc: N/A", (255, 255, 255), font=font)
            image.save(os.path.join(dir_name, "gt.png"))
            
            # save foveated
            for ecc_level in ecc_levels:
                for res_level in res_levels:
                    filename = str(ecc_level).replace(".", "_") + "___" + str(res_level).replace(".", "_")
                    image = Image.fromarray(renders[ecc_level][res_level][image_path])
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 200, 100), fill=(0, 0, 0, 180))
                    draw.text((10, 10), "Res: {:.2f}".format(res_level), (255, 255, 255), font=font)
                    draw.text((10, 50), "Ecc: {:.2f}".format(ecc_level), (255, 255, 255), font=font)
                    image.save(os.path.join(dir_name, filename + ".png"))

    results["time"]["gt"] = list(results["time"]["gt"].values())
    results["FovVideoVDP"]["gt"] = list(results["FovVideoVDP"]["gt"].values())
    os.makedirs(os.path.join(args.data_dir, "viz", args.exp_name), exist_ok=True)
    with open(os.path.join(args.data_dir, "viz", args.exp_name, "foveation_grid_search.yaml"), "w") as f:
        yaml.dump(results, f, default_flow_style=False)


def generate_heatmaps(data_dir, image_scale, res_levels, ecc_levels, time_budget, exp_name):
    colmap_data = joblib.load(os.path.join(data_dir, "train.pkl"))
    width, height = colmap_data[list(colmap_data.keys())[0]]["image_size"]
    width = width * image_scale
    height = height * image_scale
    
    data_filename = os.path.join(data_dir, "viz", exp_name, "foveation_grid_search.yaml")

    with open(data_filename) as f:
        data = yaml.safe_load(f)

    res_levels_t = res_levels[::-1]
    ecc_levels_t = ecc_levels
        
    # plot FovVideoVDP
    data_qual = np.zeros((len(res_levels_t), len(ecc_levels_t)))
    for i in range(len(res_levels_t)):
        for j in range(len(ecc_levels_t)):
            data_qual[i, j] = np.mean(data["FovVideoVDP"][ecc_levels_t[j]][res_levels_t[i]]).item()

    plt.clf()
    data_frame = pd.DataFrame(data=data_qual, columns=ecc_levels_t, index=res_levels_t)
    sns.heatmap(data_frame, square=True, annot=True, cbar_kws={"shrink": 0.5}, cmap="viridis")
    plt.title("FovVideoVDP (JOD)")
    plt.xlabel("eccentricity (degree)")
    plt.ylabel("resolution scale")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "viz", exp_name, "foveated_fovvideovdp.pdf"))
    
    # plot time
    gt_time = np.mean(data["time"]["gt"])
    data_time = np.zeros((len(res_levels_t), len(ecc_levels_t)))
    data_speedup = [[0] * len(ecc_levels_t)] * len(res_levels_t)
    for i in range(len(res_levels_t)):
        for j in range(len(ecc_levels_t)):
            data_time[i, j] = 1.0 / (np.mean(data["time"][ecc_levels_t[j]][res_levels_t[i]]).item() / gt_time)

    plt.clf()
    data_frame = pd.DataFrame(data=data_time, columns=ecc_levels_t, index=res_levels_t)
    annotations = data_frame.astype(str)
    annotations = annotations.applymap(lambda x: f"{float(x):.2f}x")
    sns.heatmap(data_frame, square=True, annot=annotations, cbar_kws={"shrink": 0.5}, cmap="viridis", fmt="s")
    plt.title("Speedup vs. Full-Resolution Rendering")
    plt.xlabel("eccentricity (degree)")
    plt.ylabel("resolution scale")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "viz", exp_name, "foveated_time.pdf"))

    # plot LOD levels
    plt.clf()
    fig, ax = plt.subplots(len(res_levels_t), len(ecc_levels_t), sharex=True, sharey=True, squeeze=False)
    categories = ["fovea", "periphery"]
    for i in range(len(res_levels_t)):
        for j in range(len(ecc_levels_t)):
            values = np.array(data["params"][ecc_levels_t[j]][res_levels_t[i]]["lod"])  # [N, 1, 2]
            mean = np.mean(values[:, 0, :], axis=0) * 100  # [N, 2]
            std = np.std(values[:, 0, :], axis=0) * 100  # [N, 2]
            ax[i, j].bar(["fov", "per"], mean, label=categories, color=["tab:blue", "tab:orange"])
            ax[i, j].bar_label(ax[i, j].containers[0], label_type="edge", fmt="%.f%%")
            ax[i, j].errorbar(["fov", "per"], mean, std, fmt=".", color="red", capsize=3)
            ax[i, j].set_yticks([])
            
            p_value = stats.ttest_rel(values[:, 0, 0], values[:, 0, 1])
            
            p_value_color = "black"
            if p_value.pvalue <= 0.05:
                p_value_color = "limegreen"
            
            if "e" in str(p_value.pvalue) or p_value.pvalue <= 0.05:
                p_value = "p={:0.2e}".format(p_value.pvalue)
            else:
                p_value = "p={:0.2f}".format(p_value.pvalue)
            
            ax[i, j].text(0.02, 0.98, p_value, horizontalalignment="left", verticalalignment="top", transform=ax[i, j].transAxes, color=p_value_color, fontsize=8)

            if j == 0:
                ax[i, j].set_ylabel(f"{res_levels_t[i]*100:g}" + "%")
            if i == len(res_levels_t) - 1:
                ax[i, j].set_xlabel(f"{ecc_levels_t[j]:g}" + "$^\circ$")

    fig.legend(categories, loc="lower right", ncols=2)

    fig.supxlabel("eccentricity")
    fig.supylabel("resolution scale")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "viz", exp_name, "foveated_lod_levels.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_gt", type=str, help="config filename", required=True)
    parser.add_argument("--config_fov", type=str, help="config filename", required=True)
    parser.add_argument("--data_dir",type=str, help="data directory", required=True)
    parser.add_argument("--display", action="store_true", help="display")
    parser.add_argument("--ecc_levels", type=float, nargs="*", help="eccentricity levels", default=[5.0, 7.5, 10.0, 15.0, 20.0])
    parser.add_argument("--eval_every", type=int, help="evaluate every N images", default=None)
    parser.add_argument("--exp_name", type=str, help="experiment_name", required=True)
    parser.add_argument("--image_scale", type=int, help="image scale", default=1)
    parser.add_argument("--input", type=str, help="LOD model filename", required=True)
    parser.add_argument("--lod_levels", type=float, nargs="*", help="LOD levels to test", default=[1.0])
    parser.add_argument("--mode", type=str, help="comparison mode [lod_ref, lod_levels]", default="lod_ref")
    parser.add_argument("--res_levels", type=float, nargs="*", help="save images", default=[0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--save_images", action="store_true", help="save images")
    parser.add_argument("--skip_grid_search", action="store_true", help="skip grid search")
    parser.add_argument("--time", type=float, help="time budget", default=None)
    parser.add_argument("--visualize", action="store_true", help="visualize optimization process")
    args = parser.parse_args()

    # grid search parameters    
    if not args.skip_grid_search:
        # profile env
        profile_env = ProfileEnv()
        profile_env.setup(mode="highest")

        test_foveation(args, args.res_levels, args.ecc_levels)
        
        profile_env.shutdown()
        
    generate_heatmaps(args.data_dir, args.image_scale, args.res_levels, args.ecc_levels, args.time, args.exp_name)

    
