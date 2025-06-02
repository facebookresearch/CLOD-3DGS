# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import List

from bayes_opt import BayesianOptimization
from colorama import Fore
import numpy as np
import pyfvvdp
import scipy
import torch

import matplotlib.pyplot as plt
from viz import visualize_optimizer, GazeHeatmapVisualizer

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


class ViewSamplerFoveation():
    """
    View sampler
    """
    def __init__(
        self,
        config,
        engine,
        fv,
        time_budget: float,
        verbose: bool=False,
        debug: bool=False,
        visualize: List[str]=[],
    ):
        self.config = config
        self.num_optim_levels = self.config.num_levels()
        self.fv = fv
            
        # parameter sampler
        self.pbounds = {}

        for i in range(self.num_optim_levels):
            self.pbounds["lod"+str(i)] = (0.05, 1.0)

        self.engine = engine
        self.time_budget = time_budget

        # parameters for debugging
        #self.verbose = verbose
        self.verbose = True
        self.debug = debug
        self.visualize = visualize
        self.iterations = 0
    

    def train(self, sample_state, res_level, gt_image, gt_time):
        # training params
        sample_params = vkgs_py.SampleParams()
        sample_params.num_frames_recorder = self.config.num_frames_recorder()
        sample_params.num_frames_benchmark = self.config.num_frames_benchmark()
        sample_params.lod = [[1.0] * self.config.num_levels()] * self.config.num_frames_recorder()
        sample_params.res = [[1.0, res_level]] * self.config.num_frames_recorder()
        sample_params.lod_params = [[vkgs_py.vec4(1.0)] * self.config.num_levels()] * self.config.num_frames_recorder()

        max_target = 0
        iteration = 0

        # set up optimizer
        def loss_function(
            **kwargs
        ):
            nonlocal max_target
            nonlocal iteration
            
            # train run
            lod = []
            for i in range(self.config.num_levels()):
                index = i % self.num_optim_levels
                lod.append(kwargs["lod"+str(index)])
            sample_params.lod = [lod] * self.config.num_frames_recorder()
            sample_params.res = [[1.0, res_level]] * self.config.num_frames_recorder()

            sample_result = self.engine.sample(sample_params, sample_state)
            data_pred = vkgs_py.get_sample_result_numpy(sample_result)
        
            fixation_point = torch.tensor([sample_state.center[0].x * self.config.res()[0],
                                           sample_state.center[0].y * self.config.res()[1]])

            value, stats = self.fv.predict(
                data_pred[0],
                gt_image,
                dim_order="HWC",
                fixation_point=fixation_point,
            )

            time_loss = 1
            quality = value.item()
            time = np.median(np.array(sample_result.time))
            if self.time_budget - time < 0:
                norm_over_time = time / self.time_budget
                time_loss = 1 / (norm_over_time**3)
            target = time_loss * quality

            if self.verbose:
                out_string = str(iteration) + "\t"
                for i in range(self.config.num_levels()):
                    index = i % self.num_optim_levels
                    out_string += str(round(lod[index], 3)) + "\t"

                out_string += str(round(quality, 3)) + "\t"
                out_string += str(round(time, 3)) + "\t"
                out_string += str(round(gt_time / time, 3)) + "\t"
                out_string += str(round(target, 3))
                if target > max_target:
                    max_target = target
                    if time <= self.time_budget:
                        print(Fore.GREEN + out_string + Fore.WHITE)
                    else:
                        print(Fore.YELLOW + out_string + Fore.WHITE)
                else:
                    print(Fore.WHITE + out_string + Fore.WHITE)

            iteration += 1
            return target

        optimizer = BayesianOptimization(
            f = loss_function,
            pbounds = self.pbounds,
            allow_duplicate_points = True,
            verbose=0
        )

        if self.verbose:
            out_string = "iter\t"
            for i in range(self.config.num_levels()):
                index = i % self.num_optim_levels                    
                out_string += "*lod" + str(index) + "\t"
    
            out_string += "qual\t"
            out_string += "time\t"
            out_string += "speedup\t"
            out_string += "loss"
            print(out_string)

        optimizer.maximize(
            init_points=10,
            n_iter=10,
        )

        max_lod_keys = sorted([x for x in optimizer.max["params"] if x.startswith("lod")])
        max_lod = [optimizer.max["params"][x] for x in max_lod_keys]
        max_res_keys = sorted([x for x in optimizer.max["params"] if x.startswith("res")])
        max_res = [optimizer.max["params"][x] for x in max_res_keys]
        print("Target:", optimizer.max["target"], "\tLOD:", max_lod)
        print("Target:", optimizer.max["target"], "\tRes:", max_res)

        output = {}
        output["target"] = optimizer.max["target"]
        output["params"] = vkgs_py.SampleParams()
        output["params"].num_frames_recorder = self.config.num_frames_recorder()
        output["params"].num_frames_benchmark = self.config.num_frames_benchmark()        
        output["params"].lod = [max_lod] * self.config.num_frames_recorder()
        output["params"].res = [[1.0, res_level]] * self.config.num_frames_recorder()
        output["params"].lod_params = [[vkgs_py.vec4(1.0)] * self.config.num_levels()] * self.config.num_frames_recorder()

        # visualize
        if "samples" in self.visualize:
            visualize_optimizer(optimizer)

        if "render" in self.visualize:
            sample_result = self.engine.sample(output["params"], sample_state)
            data_pred = vkgs_py.get_sample_result_numpy(sample_result)

            plt.clf()
            plt.imshow(data_pred[0])
            plt.show()

        self.iterations += 1
        return output