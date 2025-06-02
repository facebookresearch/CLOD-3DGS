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

from metrics import get_pyfvvdp_metric


def test_videos(gt_dir, pred_dir):
    image = Image.open(os.path.join(gt_dir, "frame_0.png"))
    width, height = image.size
    
    device = torch.device("cuda:0")
    
    fv = get_pyfvvdp_metric(
        width=width,
        height=height,
        mode="standard_fhd",
        foveated=True,
        device=device,
    )

    errors = []
    progress = tqdm(total=1200)
    for i in range(0, 1200, 120):
        gt_tensor = torch.zeros((height, width, 3, 120))
        pred_tensor = torch.zeros((height, width, 3, 120))

        for f in range(0, 120):
            index = i + f
            gt_image = np.array(Image.open(os.path.join(gt_dir, "frame_0.png")), dtype="float32") / 255.0
            pred_image = np.array(Image.open(os.path.join(pred_dir, "frame_0.png")), dtype="float32") / 255.0
            
            gt_tensor[..., f] = torch.from_numpy(gt_image).to(device)
            pred_tensor[..., f] = torch.from_numpy(pred_image).to(device)
            progress.update(1)

        value, _ = fv.predict(pred_tensor, gt_tensor, dim_order="HWCF", frames_per_second=60)
        errors.append(value.item())
    progress.close()

    print(pred_dir + ":", np.mean(errors).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, help="ground truth directory", required=True)
    parser.add_argument("--pred_dir", type=str, help="predicted directory", required=True)
    args = parser.parse_args()

    test_videos(args.gt_dir, args.pred_dir)

    
