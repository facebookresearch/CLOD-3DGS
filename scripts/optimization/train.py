# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from subprocess import list2cmdline
import sys
import time

from bayes_opt import BayesianOptimization
import numpy as np
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import List

from profile_env import ProfileEnv
from trainer import Trainer

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config filename", required=True)
    parser.add_argument("--data_dir",type=str, help="data directory", required=True)
    parser.add_argument("--default_lod_params", type=float, nargs=4, help="fixed LOD params", default=[1.0, 1.0, 1.0, 1.0])
    parser.add_argument("--display", action="store_true", help="display")
    parser.add_argument("--distance_lod", action="store_true", help="use distance LOD")
    parser.add_argument("--dynamic_res", action="store_true", help="dyamic resolution")
    parser.add_argument("--image_dir", type=str, help="image directory")
    parser.add_argument("--image_scale", type=int, help="image scale", default=1)
    parser.add_argument("--input_lod", type=str, help="LOD model filename", required=True)
    parser.add_argument("--input_ref", type=str, help="reference model filename", required=True)
    parser.add_argument("--model_name", type=str, help="model name", required=True)
    parser.add_argument("--nn_mode", type=str, help="neural network mode", default="time")
    parser.add_argument("--num_gaze_dir", type=int, help="number of gaze directions", default=0)
    parser.add_argument("--single_level", action="store_true", help="single level")
    parser.add_argument("--skip_active", action="store_true", help="skip active learning training")
    parser.add_argument("--skip_init", action="store_true", help="skip initial COLMAP training")
    parser.add_argument("--skip_nn", action="store_true", help="skip neural network training")
    parser.add_argument("--skip_time", action="store_true", help="skip time sampling")
    parser.add_argument("--time_budget", type=float, help="time budget (in milliseconds)", default=1.0)
    parser.add_argument("--visualize", nargs="*", type=str, help="visualize optimization process", default=[])
    parser.add_argument("--verbose", action="store_true", help="printing loss information")
    args = parser.parse_args()
    
    # save command line arguments
    os.makedirs(os.path.join(args.data_dir, "models", args.model_name), exist_ok=True)
    command_line = list2cmdline(["python"] + sys.argv)
    with open(os.path.join(args.data_dir, "models", args.model_name, "cmd.txt"), "w") as f:
        f.write(command_line)

    # profile env
    profile_env = ProfileEnv()
    profile_env.setup(mode="highest")
    
    # parameters
    num_frames_benchmark = 12
    num_frames_recorder = 1
    
    files = os.listdir(args.image_dir)
    image = Image.open(os.path.join(args.image_dir, files[0]))
    width = image.size[0] * args.image_scale
    height = image.size[1] * args.image_scale

    print("Render resolution: (" + str(width) + ", " + str(height) + ")")

    config = vkgs_py.Config(args.config, "immediate")
    config.num_frames_benchmark(num_frames_benchmark)
    config.num_frames_recorder(num_frames_recorder)
    config.res(width, height);
    config.dynamic_res(args.dynamic_res);

    config_gt = vkgs_py.Config("../../config/full.yaml", "immediate")
    config_gt.num_frames_benchmark(num_frames_benchmark)
    config_gt.num_frames_recorder(num_frames_recorder)
    config_gt.res(width, height);
    config_gt.dynamic_res(False);

    scene_name = os.path.normpath(args.data_dir).split(os.sep)[-1]
    experiment_name = scene_name + "___" + args.model_name + "___" + str(int(time.time() * 1000))
    writer = SummaryWriter(log_dir=os.path.join("../../runs/", experiment_name))

    # create trainer
    trainer = Trainer(
        filename_input=args.input_lod,
        filename_gt=args.input_ref,
        config_input=config,
        config_gt=config_gt,
        time_budget=args.time_budget,
        verbose=args.verbose,
        debug=False,
        visualize=args.visualize,
        image_dir=args.image_dir,
        data_dir=args.data_dir,
        model_name=args.model_name,
        num_gaze_dir=args.num_gaze_dir,
        distance_lod=args.distance_lod,
        single_level=args.single_level,
        default_lod_params=args.default_lod_params,
    )

    # train on COLMAP data
    if not args.skip_init:
        for iterations in range(trainer.get_num_samples_colmap()):
            sample_state = trainer.get_sample_colmap()
            trainer.sample_iteration(sample_state, "COLMAP")
    else:
        print("Skipped initial training")

    # load initial training data
    trainer.dataset_train.load_data(os.path.join(args.data_dir, "models", args.model_name, "data.yaml"))
    trainer.dataset_val.load_data(os.path.join(args.data_dir, "models", args.model_name, "data.yaml"))
    
    # search for performance areas
    if not args.skip_time:
        for i in range(10):
            trainer.get_sample_stratisfied(sequence_length=5)
            trainer.dataset_time.save(os.path.join(args.data_dir, "models", args.model_name, "time.json"))
            trainer.dataset_time.save(os.path.join(args.data_dir, "models", args.model_name, "time.yaml"))

    if args.nn_mode == "time":
        trainer.dataset_train.load_data(os.path.join(args.data_dir, "models", args.model_name, "time.json"))
        trainer.dataset_val.load_data(os.path.join(args.data_dir, "models", args.model_name, "time.json"))

    # perform initial training
    if not args.skip_nn:
        trainer.dataset_val.save(os.path.join(args.data_dir, "models", args.model_name, "val.yaml"))
        for i in range(100_000):
            loss = trainer.train_epoch()["loss"]
            writer.add_scalar("train/loss", loss, i)
            if (i+1) % 10 == 0:
                loss = trainer.validate_epoch()["loss"]
                writer.add_scalar("valid/loss", loss, i)
                trainer.save_model(args.model_name)
    else:
        print("Skipped neural network training")

    trainer.load_model(args.model_name)

    # search for areas to improve generalization
    if not args.skip_active:
        max_iterations = 1_000
        for iteration in range(max_iterations):
            # step 1: explore a few samples (look for off-budget predictions)
            sample_state = trainer.get_sample_active_learning(sequence_length=10, num_samples=10)

            # step 2: train the worst sample
            trainer.sample_iteration(sample_state, "active")
            
            # step 3: retrain neural network periodically
            if iteration % 10 == 9:
                for i in range(1000):
                    loss = trainer.train_epoch()["loss"]
                    writer.add_scalar("train/loss", loss, i)
                    if (i+1) % 10 == 0:
                        loss = trainer.validate_epoch()["loss"]
                        writer.add_scalar("valid/loss", loss, i)
                trainer.save_model(args.model_name)
    else:
        print("Skipped active learning phase")

    writer.close()

    trainer.shutdown()
    profile_env.shutdown()