# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

import joblib
import yaml

from colmap import convert_colmap_camera, get_gaze_dir_global

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


def convert_pkl_to_yaml(data_dir):
    config = vkgs_py.Config("../../config/full.yaml", "immediate")
    config.num_frames_benchmark(1)
    config.num_frames_recorder(1)
    
    scenes = os.listdir(data_dir)
    for scene in scenes:
        filename_input = os.path.join(data_dir, scene, "ckpts", "ckpt_59999_rank0.ply")

        engine = vkgs_py.Engine(config)
        engine.load_splats(filename_input)
        engine.start()
        model_transform = vkgs_py.mat4_to_numpy(engine.get_model_matrix())

        for split in ["train", "val"]:
            colmap_data = joblib.load(os.path.join(data_dir, scene, split + ".pkl"))
            data = {}

            index = 0
            for key, view_params in colmap_data.items():
                data[index] = {}

                z_near = 0.01
                z_far = 100.0
                data_sample = convert_colmap_camera(view_params, model_transform, z_near, z_far)
                
                data[index] = {"sample_state": {}, "metadata": {}}

                data[index]["sample_state"]["pos"] = [[pos.x, pos.y, pos.z] for pos in data_sample["sample_state"].pos]
                data[index]["sample_state"]["quat"] = [[quat.w, quat.x, quat.y, quat.z] for quat in data_sample["sample_state"].quat]
                data[index]["sample_state"]["view_angles"] = [[v.angle_right, v.angle_left, v.angle_down, v.angle_up] for v in data_sample["sample_state"].view_angles]
                data[index]["sample_state"]["gaze_dir"] = [[0.0, 0.0, 0.0]]
                data[index]["sample_state"]["center"] = [[0.0, 0.0]]

                data[index]["metadata"]["filename"] = key
                index += 1

            with open(os.path.join(data_dir, scene, split + ".yaml"), "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        engine.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir")
    args = parser.parse_args()

    convert_pkl_to_yaml(args.data_dir)
