# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
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


def test_lod_levels(args, config, engines, colmap_data, metric_funcs):
    sample_params = vkgs_py.SampleParams()
    sample_params.num_frames_benchmark = num_frames_benchmark
    sample_params.num_frames_recorder = num_frames_recorder
    sample_params.lod = [[1.0] * config.num_levels()] * num_frames_recorder
    sample_params.res = [[1.0] * config.num_levels()] * num_frames_recorder
    sample_params.lod_params = [[vkgs_py.vec4(1.0)] * config.num_levels()] * num_frames_recorder

    renders = {}
    times = {}

    # create output directory to store results
    scene_name = Path(args.data_dir).parts[-1]
    output_dir = os.path.join("../../results/lod", args.model_type, scene_name)
    os.makedirs(output_dir, exist_ok=True)
    
    progress = tqdm(total=len(colmap_data))

    metrics = {}
    if "fv" in metric_funcs:
        metrics["FovVideoVDP"] = {}
    if "psnr" in metric_funcs:
        metrics["PSNR"] = {}
    if "ssim" in metric_funcs:
        metrics["SSIM"] = {}
    if "lpips" in metric_funcs:
        metrics["LPIPS"] = {}
    if "time" in metric_funcs:
        metrics["Time"] = {}

    time_metric = {}
    for metric in metrics.keys():
        for lod_level in args.lod_levels:
            metrics[metric]["lod_" + str(lod_level)] = []
    for lod_level in args.lod_levels:
        time_metric["lod_" + str(lod_level)] = []

    for image_path, view_params in colmap_data.items():
        renders.clear()
        times.clear()
        
        for lod_level in args.lod_levels:
            sample_params.lod = [[lod_level] * config.num_levels()] * num_frames_recorder
            model_transform = vkgs_py.mat4_to_numpy(engines["lod"].get_model_matrix())
            sample_state = convert_colmap_camera(view_params, model_transform, z_near, z_far)["sample_state"]
            sample_state.center = [vkgs_py.vec2(0.5)] * num_frames_recorder

            sample_result = engines["lod"].sample(sample_params, sample_state)
            data_pred = vkgs_py.get_sample_result_numpy(sample_result)
            time_metric["lod_" + str(lod_level)].append(sample_result.time)

            filename = os.path.join(args.image_dir, view_params["filename"])
            gt_image = np.array(Image.open(filename))

            render = np.array(data_pred[0])
            renders["lod_" + str(lod_level)] = render
            times["lod_" + str(lod_level)] = np.median(sample_result.time).item()

        # plot figures
        if args.visualize:
            plt.clf()
            plt.close()
            fig, ax = plt.subplots(1, len(args.lod_levels) + 1, sharex=True, sharey=True)
            for i in range(len(args.lod_levels)):
                lod_level = args.lod_levels[i]
                ax[i].set_title("LOD:" + str(lod_level*100) + "%")
                ax[i].imshow(renders["lod_" + str(lod_level)])
                if (i > 0):
                    ax[i].get_yaxis().set_ticks([])

            ax[len(args.lod_levels)].set_title("ground truth")
            ax[len(args.lod_levels)].imshow(gt_image)
            ax[len(args.lod_levels)].get_yaxis().set_ticks([])
            fig.suptitle(view_params["filename"])
            fig.tight_layout()
            plt.show()

        # save images
        filename = os.path.splitext(view_params["filename"])[0] + ".png"
        os.makedirs(os.path.join(output_dir, "gt"), exist_ok=True)
        Image.open(os.path.join(args.image_dir, view_params["filename"])).save(os.path.join(output_dir, "gt", filename))
        for key in renders.keys():
            dir_name = os.path.join(output_dir, key.replace(".", "_"))
            os.makedirs(dir_name, exist_ok=True)
            Image.fromarray(renders[key]).save(os.path.join(dir_name, filename))

        # compute metrics [FovVideoVDP]
        for lod_level in args.lod_levels:
            if "fv" in metric_funcs:
                value, stats = metric_funcs["fv"].predict(renders["lod_" + str(lod_level)], gt_image, dim_order="HWC")
                metrics["FovVideoVDP"]["lod_" + str(lod_level)].append(value.item())

        # compute metrics [PSNR, SSIM, LPIPS]
        for lod_level in args.lod_levels:
            t_lod = im2tensor(renders["lod_"+str(lod_level)]).to(torch.device("cuda", 0))
            t_gt = im2tensor(gt_image).to(torch.device("cuda", 0))
            
            if "psnr" in metric_funcs:
                metrics["PSNR"]["lod_" + str(lod_level)].append(metric_funcs["psnr"](t_lod, t_gt).item())
            if "ssim" in metric_funcs:
                metrics["SSIM"]["lod_" + str(lod_level)].append(metric_funcs["ssim"](t_lod, t_gt).item())
            if "lpips" in metric_funcs:
                metrics["LPIPS"]["lod_" + str(lod_level)].append(metric_funcs["lpips"](t_lod, t_gt).item())
            if "time" in metric_funcs:
                metrics["Time"]["lod_" + str(lod_level)].append(times["lod_" + str(lod_level)])

        progress.update(1)
    progress.close()

    # save and print metrics results
    with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
        f.write("Metric," + ",".join([str(x) for x in args.lod_levels]) + "\n")
        for metric in metrics:
            row = []
            row.append(metric)
            print("Metric:", metric)
            for lod_level in args.lod_levels:
                metrics[metric]["lod" + str(lod_level)] = np.array(metrics[metric]["lod_" + str(lod_level)])
                print("\tlod" + str(lod_level) + ":\t", str(np.mean(metrics[metric]["lod_" + str(lod_level)])) + "\t", str(np.std(metrics[metric]["lod_" + str(lod_level)])))
                row.append(str(np.mean(metrics[metric]["lod_" + str(lod_level)])))

            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config filename", required=True)
    parser.add_argument("--data_dir",type=str, help="data directory", required=True)
    parser.add_argument("--display", action="store_true", help="display")
    parser.add_argument("--image_dir", type=str, help="image directory", required=True)
    parser.add_argument("--input_lod", type=str, help="LOD model filename", required=True)
    parser.add_argument("--heatmap", action="store_true", help="display heatmap")
    parser.add_argument("--lod_levels", type=float, nargs="*", help="LOD levels to test", default=[1.0, 0.5, 0.1, 0.05])
    parser.add_argument("--metrics", type=str, default=["fv", "psnr", "ssim", "lpips", "time"])
    parser.add_argument("--model_type", type=str, help="model type", required=True)
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

    # load ground truth data
    engines = {}

    model_names = {"lod": args.input_lod}
    
    for model in list(model_names.keys()):
        engines[model] = vkgs_py.Engine(config, False)
        engines[model].load_splats(model_names[model])
        engines[model].start()

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
    for split in ["val"]:
        colmap_data = joblib.load(os.path.join(args.data_dir, split + ".pkl"))
        test_lod_levels(args, config, engines, colmap_data, metric_funcs)

    profile_env.shutdown()
    
