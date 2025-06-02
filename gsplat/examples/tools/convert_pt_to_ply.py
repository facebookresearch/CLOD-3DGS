# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch

from utils_io import save_ply


def convert_pt_to_ply(filename):
    data = torch.load(filename, weights_only=True)["splats"]

    output = {}
    output["x"] = data["means"][..., 0].detach().cpu().numpy()
    output["y"] = data["means"][..., 1].detach().cpu().numpy()
    output["z"] = data["means"][..., 2].detach().cpu().numpy()

    output["f_dc_0"] = data["sh0"][..., 0, 0].detach().cpu().numpy()
    output["f_dc_1"] = data["sh0"][..., 0, 1].detach().cpu().numpy()
    output["f_dc_2"] = data["sh0"][..., 0, 2].detach().cpu().numpy()

    for i in range(data["shN"].shape[1]):
        i0 = (0 * data["shN"].shape[1]) + i
        i1 = (1 * data["shN"].shape[1]) + i
        i2 = (2 * data["shN"].shape[1]) + i
        output["f_rest_" + str(i0)] = data["shN"][..., i, 0].detach().cpu().numpy()
        output["f_rest_" + str(i1)] = data["shN"][..., i, 1].detach().cpu().numpy()
        output["f_rest_" + str(i2)] = data["shN"][..., i, 2].detach().cpu().numpy()

    output["scale_0"] = data["scales"][..., 0].detach().cpu().numpy()
    output["scale_1"] = data["scales"][..., 1].detach().cpu().numpy()
    output["scale_2"] = data["scales"][..., 2].detach().cpu().numpy()

    output["rot_0"] = data["quats"][..., 0].detach().cpu().numpy()
    output["rot_1"] = data["quats"][..., 1].detach().cpu().numpy()
    output["rot_2"] = data["quats"][..., 2].detach().cpu().numpy()
    output["rot_3"] = data["quats"][..., 3].detach().cpu().numpy()

    output["opacity"] = data["opacities"].detach().cpu().numpy()

    save_ply(filename.replace(".pt", ".ply"), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="input filename")
    args = parser.parse_args()

    convert_pt_to_ply(args.src)
