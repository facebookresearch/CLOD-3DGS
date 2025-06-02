# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
from pathlib import Path
import sys

from bayes_opt import BayesianOptimization
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyfvvdp
import scipy
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

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


def test_distance_lod(args, config, colmap_data, metric_funcs):
    sample_params = vkgs_py.SampleParams()
    sample_params.num_frames_benchmark = num_frames_benchmark
    sample_params.num_frames_recorder = num_frames_recorder
    sample_params.lod = [[1.0] * config.num_levels()] * num_frames_recorder
    sample_params.res = [[1.0] * config.num_levels()] * num_frames_recorder
    sample_params.lod_params = [[vkgs_py.vec4(1.0)] * config.num_levels()] * num_frames_recorder

    renders = {"gt": {}}

    # create output directory to store results
    scene_name = Path(args.data_dir).parts[-1]
    output_dir = os.path.join("../../results/dist_lod", scene_name)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {}
    if "psnr" in metric_funcs:
        metrics["PSNR"] = {"gt": {"": 0}}
    if "ssim" in metric_funcs:
        metrics["SSIM"] = {"gt": {"": 0}}
    if "lpips" in metric_funcs:
        metrics["LPIPS"] = {"gt": {"": 0}}
    if "time" in metric_funcs:
        metrics["Time"] = {"gt": {"": 0}}

    # populate initial dictionaries
    for metric in metrics.keys():
        for lod_distance in args.lod_distances:
            metrics[metric][lod_distance] = {}
            for lod_percentage in args.lod_percentages:
                metrics[metric][lod_distance][lod_percentage] = {}
    for lod_distance in args.lod_distances:
        renders[lod_distance] = {}
        for lod_percentage in args.lod_percentages:
            renders[lod_distance][lod_percentage] = {}

    # load ground truth data
    engine = vkgs_py.Engine(config, False)
    engine.load_splats(args.input)
    engine.start()

    # setup sample params
    sample_params = vkgs_py.SampleParams()
    sample_params.num_frames_benchmark = num_frames_benchmark
    sample_params.num_frames_recorder = num_frames_recorder
    sample_params.lod = [[1.0] * config.num_levels()] * num_frames_recorder
    sample_params.res = [[1.0] * config.num_levels()] * num_frames_recorder
    sample_params.lod_params = [[vkgs_py.vec4(1.0)] * config.num_levels()] * num_frames_recorder
    
    # render ground truth
    num_images = len(colmap_data)
    if args.eval_every is not None:
        num_images = math.ceil(num_images / args.eval_every)

    progress = tqdm(total=num_images)
    index = 0
    for image_path, view_params in colmap_data.items():
        if args.eval_every and index % args.eval_every:
            index += 1
            continue

        model_transform = vkgs_py.mat4_to_numpy(engine.get_model_matrix())
        sample_state = convert_colmap_camera(view_params, model_transform, z_near, z_far)["sample_state"]
        sample_state.center = [vkgs_py.vec2(0.5)] * num_frames_recorder        

        sample_result = engine.sample(sample_params, sample_state)
        data_pred = vkgs_py.get_sample_result_numpy(sample_result)
        renders["gt"][image_path] = np.array(data_pred[0])

        metrics["Time"]["gt"][image_path] = np.median(sample_result.time).item()

        filename = os.path.splitext(view_params["filename"])[0] + ".png"
        os.makedirs(os.path.join(output_dir, "gt"), exist_ok=True)
        Image.fromarray(renders["gt"][image_path]).save(os.path.join(output_dir, "gt", filename))

        index += 1
        progress.update(1)
    progress.close()

    # grid search
    progress = tqdm(total=num_images)
    index = 0
    for image_path, view_params in colmap_data.items():
        if args.eval_every and index % args.eval_every:
            index += 1
            continue

        for lod_distance in args.lod_distances:
            if lod_distance not in renders:
                renders[lod_distance] = {}
                metrics["Time"][lod_distance] = {}

            for lod_percentage in args.lod_percentages:
                if lod_percentage not in renders[lod_distance]:
                    renders[lod_distance][lod_percentage] = {}
                    metrics["Time"][lod_distance][lod_percentage] = {}
                
                sample_params.lod = [[1.0] * config.num_levels()] * num_frames_recorder
                sample_params.lod_params = [[vkgs_py.vec4(lod_percentage, 1.0, 1.0, lod_distance)] * config.num_levels()] * num_frames_recorder
                model_transform = vkgs_py.mat4_to_numpy(engine.get_model_matrix())
                sample_state = convert_colmap_camera(view_params, model_transform, z_near, z_far)["sample_state"]
                sample_state.center = [vkgs_py.vec2(0.5)] * num_frames_recorder

                sample_result = engine.sample(sample_params, sample_state)
                data_pred = vkgs_py.get_sample_result_numpy(sample_result)
                metrics["Time"][lod_distance][lod_percentage][image_path] = np.median(sample_result.time).item()

                render = np.array(data_pred[0])
                renders[lod_distance][lod_percentage][image_path] = render

                filename = os.path.splitext(view_params["filename"])[0] + ".png"
                exp_name = str(lod_distance).replace(".", "_") + "___" + str(lod_percentage).replace(".", "_")
                os.makedirs(os.path.join(output_dir, exp_name), exist_ok=True)
                Image.fromarray(renders[lod_distance][lod_percentage][image_path]).save(os.path.join(output_dir, exp_name, filename))

        index += 1
        progress.update(1)
    progress.close()

    engine.end()

    index = 0
    for image_path in renders["gt"].keys():
        # compute metrics [PSNR, SSIM, LPIPS]
        for lod_distance in args.lod_distances:
            if lod_distance not in metrics["PSNR"]:
                metrics["PSNR"][lod_distance] = {}
                metrics["SSIM"][lod_distance] = {}
                metrics["LPIPS"][lod_distance] = {}

            for lod_percentage in args.lod_percentages:
                t_lod = im2tensor(renders[lod_distance][lod_percentage][image_path]).to(torch.device("cuda", 0))
                t_gt = im2tensor(renders["gt"][image_path]).to(torch.device("cuda", 0))

                if lod_percentage not in metrics["PSNR"][lod_distance]:
                    metrics["PSNR"][lod_distance][lod_percentage] = {}
                    metrics["SSIM"][lod_distance][lod_percentage] = {}
                    metrics["LPIPS"][lod_distance][lod_percentage] = {}

                if "psnr" in metric_funcs:
                    metrics["PSNR"][lod_distance][lod_percentage][image_path] = metric_funcs["psnr"](t_lod, t_gt).item()
                if "ssim" in metric_funcs:
                    metrics["SSIM"][lod_distance][lod_percentage][image_path] = metric_funcs["ssim"](t_lod, t_gt).item()
                if "lpips" in metric_funcs:
                    metrics["LPIPS"][lod_distance][lod_percentage][image_path] = metric_funcs["lpips"](t_lod, t_gt).item()

        progress.update(1)
    progress.close()

    # save and print metrics results
    exp_names = ["gt"]
    for lod_distance in args.lod_distances:
        for lod_percentage in args.lod_percentages:
            exp_names.append(str(lod_distance).replace(".", "_") + "___" + str(lod_percentage).replace(".", "_"))

    with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
        f.write("Metric," + ",".join(exp_names) + "\n")
        for metric in metrics:
            row = []
            row.append(metric)
            print("Metric:", metric)
            
            # append ground truth
            values = []
            for image_path in metrics[metric]["gt"].keys():
                values.append(metrics[metric]["gt"][image_path])
            value = np.mean(values).item()
            row.append(str(value))
            
            # append ablation results
            for lod_distance in args.lod_distances:
                for lod_percentage in args.lod_percentages:
                    exp_name = str(lod_distance).replace(".", "_") + "___" + str(lod_percentage).replace(".", "_")
                    values = []
                    for image_path in metrics[metric][lod_distance][lod_percentage].keys():
                        values.append(metrics[metric][lod_distance][lod_percentage][image_path])
                    value = np.mean(values).item()
                    row.append(str(value))

            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config filename", required=True)
    parser.add_argument("--data_dir",type=str, help="data directory", required=True)
    parser.add_argument("--display", action="store_true", help="display")
    parser.add_argument("--eval_every", type=int, help="evaluate every N images", default=None)
    parser.add_argument("--lod_distances", type=float, nargs="+", help="LOD distances", default=[2.5, 5.0, 7.5, 10.0])
    parser.add_argument("--lod_percentages", type=float, nargs="+", help="LOD percentages", default=[0.75, 0.5, 0.25, 0.1, 0.05])
    parser.add_argument("--image_dir", type=str, help="image directory", required=True)
    parser.add_argument("--input", type=str, help="model filename", required=True)
    parser.add_argument("--metrics", type=str, default=["psnr", "ssim", "lpips", "time"])
    parser.add_argument("--visualize", action="store_true", help="visualize optimization process")
    args = parser.parse_args()
    
    # profile env
    profile_env = ProfileEnv()
    profile_env.setup(mode="highest")
        
    # load sample image to get image size (assumes all images are the same size)
    files = os.listdir(args.image_dir)
    image = Image.open(os.path.join(args.image_dir, files[0]))
    width, height = image.size

    config = vkgs_py.Config(args.config, "immediate")
    config.num_frames_benchmark(num_frames_benchmark)
    config.num_frames_recorder(num_frames_recorder)
    config.res(width, height);
    config.dynamic_res(False);

    metric_funcs = {}
    if "fv" in args.metrics:
        metric_funcs["fv"] = get_pyfvvdp_metric(
            width,
            height,
            "standard_fhd",
            foveated=False,
            device=torch.device("cuda", 0),
        )
    if "psnr" in args.metrics:
        metric_funcs["psnr"] = PeakSignalNoiseRatio(data_range=(0, 1)).to(torch.device("cuda", 0))
    if "ssim" in args.metrics:
        metric_funcs["ssim"] = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(torch.device("cuda", 0))
    if "lpips" in args.metrics:
        metric_funcs["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(torch.device("cuda", 0))
    if "time" in args.metrics:
        metric_funcs["time"] = None

    data = {}
    for split in ["train"]:
        colmap_data = joblib.load(os.path.join(args.data_dir, split + ".pkl"))
        test_distance_lod(args, config, colmap_data, metric_funcs)

    profile_env.shutdown()
    
