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

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


def im2tensor(img):
    output = img / 255.0
    output = torch.tensor(output).float()
    output = output.permute(2, 0, 1).unsqueeze(0)
    return output


def find_image(path):
    images = os.listdir(os.path.join(Path(path).parent))
    for image in images:
        if Path(image).stem == Path(path).stem:
            return os.path.join(Path(path).parent, image)
    
    raise ValueError("Could not find image: ", path)


def test_image_folders(args):
    # initialize metric functions
    metric_funcs = {}
    metric_funcs["psnr"] = PeakSignalNoiseRatio(data_range=(0, 1)).to(torch.device("cuda", 0))
    metric_funcs["ssim"] = StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(torch.device("cuda", 0))
    metric_funcs["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(torch.device("cuda", 0))

    # allocate metrics
    metrics = {}

    # decode method paths
    methods_dirs = {}
    for i in range(0, len(args.dirs), 2):
        methods_dirs[args.dirs[i]] = args.dirs[i+1]
        metrics[args.dirs[i]] = {
            "PSNR": [],
            "SSIM": [],
            "LPIPS": [],
        }

    image_names = os.listdir(methods_dirs[list(methods_dirs.keys())[0]])
    image_names = [Path(x).stem for x in image_names if Path(x).suffix.lower() in [".png", ".jpg", ".jpeg"]]
    print("Ground truth path:", args.dir_gt)

    device = torch.device("cuda", 0)
    
    # load ground truth images
    progress = tqdm(total=len(image_names))
    for image_name in image_names:
        t_gt = im2tensor(np.array(Image.open(find_image(os.path.join(args.dir_gt, image_name))))).to(device)
        for method, path in methods_dirs.items():
            t_pred = im2tensor(np.array(Image.open(find_image(os.path.join(path, image_name)))))[:, :3].to(device)
            for metric, metric_func in metric_funcs.items():
                metrics[method][metric.upper()].append(metric_func(t_pred, t_gt).item())
            
        progress.update(1)
    progress.close()

    # average results
    for method in methods_dirs.keys():
        print("Method:", method)
        for metric in metric_funcs.keys():
            metrics[method][metric.upper()] = np.mean(metrics[method][metric.upper()]).item()
            print("\t" + metric + ":", metrics[method][metric.upper()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_gt", type=str, help="ground truth directory", required=True)
    parser.add_argument("--dirs", type=str, nargs="+", help="directories (name, path)", required=True)
    args = parser.parse_args()
    
    test_image_folders(args)
    
