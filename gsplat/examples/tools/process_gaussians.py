# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np

from gaussians import Gaussians
from utils_io import read_ply, save_ply


def process_gaussians(
    src_filename,
    dst_filename,
    limit=None,
    scale=1.0,
    subsample=1.0,
    subsample_mode="basic",
):
    print("Reading input Gaussians")
    gaussian_params = read_ply(src_filename)

    # create Gaussian object
    gaussians = Gaussians(gaussian_params)

    # perform processing
    if subsample < 1.0:
        if subsample_mode == "basic":
            gaussians.subsample_basic(subsample)
        elif subsample_mode == "size":
            gaussians.subsample_size(subsample)
        print("Subsample Gaussians")

    if scale != 1.0:
        gaussians.scale_gaussians(scale)
        print("Scale Gaussians")

    if limit is not None:
        gaussians.limit(limit)
        print("Limit Gaussians")

    print("Saving output Gaussians")
    gaussian_params = gaussians.get_params()
    save_ply(dst_filename, gaussian_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downsample", type=int, help="downsample Gassians by a factor", default=2)
    parser.add_argument("--dst", type=str, help="destination filename", required=True)
    parser.add_argument("--limit", type=int, help="limit number of Gaussians", default=None)
    parser.add_argument("--scale", type=float, help="scale factor for artificially increasing Gaussian size", default=1.0)
    parser.add_argument("--subsample", type=float, help="subsample Gaussians by a factor", default=1.0)
    parser.add_argument("--subsample_mode", type=str, help="subsampling algorithm", choices=["basic", "size"], default="basic")
    parser.add_argument("--src", type=str, help="source filename", required=True)
    args = parser.parse_args()

    process_gaussians(
        src_filename=args.src,
        dst_filename=args.dst,
        limit=args.limit,
        scale=args.scale,
        subsample=args.subsample,
        subsample_mode=args.subsample_mode,
    )
