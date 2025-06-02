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


def count_gaussians(
    src_filename,
    stats = False,
):
    print("Reading input Gaussians")
    gaussian_params = read_ply(src_filename)
    print("# of Gaussians:", len(gaussian_params["x"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source filename", required=True)
    args = parser.parse_args()

    count_gaussians(
        src_filename=args.src,
    )
