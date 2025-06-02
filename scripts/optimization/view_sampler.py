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


class ViewSampler():
    """
    View sampler
    """
    def __init__(
        self,
        config,
        engine,
        engine_gt,
        time_budget: float,
        verbose: bool=False,
        debug: bool=False,
        visualize: List[str]=[],
        num_gaze_dir: int=0,
        distance_lod: bool=False,
        single_level: bool=False,
        default_lod_params: List[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        self.config = config
        self.distance_lod = distance_lod
        self.single_level = single_level
        self.default_lod_params = default_lod_params

        if single_level:
            self.num_optim_levels = 1
        else:
            self.num_optim_levels = self.config.num_levels()
            
        # parameter sampler
        self.pbounds = {}

        if not self.distance_lod:
            for i in range(self.num_optim_levels):
                self.pbounds["lod"+str(i)] = (0.05, 1.0)
        else:
            for i in range(self.num_optim_levels):
                self.pbounds["min_lod"+str(i)] = (0.05, 1.0)
                self.pbounds["max_lod"+str(i)] = (0.05, 1.0)
                self.pbounds["min_dist"+str(i)] = (1.0, 1.0)
                self.pbounds["max_dist"+str(i)] = (5.0, 5.0)

        if self.config.dynamic_res():
            for i in range(self.num_optim_levels):
                self.pbounds["res"+str(i)] = (1.0, 1.0)

        self.engine = engine
        self.engine_gt = engine_gt
        self.time_budget = time_budget

        # parameters for debugging
        self.verbose = verbose
        self.debug = debug
        self.visualize = visualize
        self.num_gaze_dir = num_gaze_dir
        self.iterations = 0
        if "gaze_heatmap" in self.visualize:
            self.gaze_heatmap_visualizer = GazeHeatmapVisualizer(self.num_gaze_dir)

        # set up metrics
        # based on "standard_fhd" configuration
        display_geometry = pyfvvdp.fvvdp_display_geometry(
            resolution = [config.res()[0], config.res()[1]],
            distance_m = 0.6,
            diagonal_size_inches = 24,
        )
        display_photometry = pyfvvdp.fvvdp_display_photo_eotf(
            contrast = 1000,
            E_ambient = 250,
            Y_peak = 200,
        )
        self.fv = pyfvvdp.fvvdp(
            display_geometry=display_geometry,
            display_photometry=display_photometry,
            device=torch.device("cpu"),
            foveated=True
        )
    

    def train(self, sample_state):
        # ground truth params
        sample_params = vkgs_py.SampleParams()
        sample_params.num_frames_recorder = self.config.num_frames_recorder()
        sample_params.num_frames_benchmark = self.config.num_frames_benchmark()
        sample_params.lod = [[1.0] * self.config.num_levels()] * self.config.num_frames_recorder()
        sample_params.res = [[1.0] * self.config.num_levels()] * self.config.num_frames_recorder()
        sample_params.lod_params = [[vkgs_py.vec4(*self.default_lod_params)] * self.config.num_levels()] * self.config.num_frames_recorder()

        # ground truth data
        sample_result = self.engine_gt.sample(sample_params, sample_state)
        sample_result = self.engine_gt.sample(sample_params, sample_state)
        data_gt = vkgs_py.get_sample_result_numpy(sample_result)
        gt_time = np.median(np.array(sample_result.time))
        print("GT time:", gt_time)

        # training params
        sample_params = vkgs_py.SampleParams()
        sample_params.num_frames_recorder = self.config.num_frames_recorder()
        sample_params.num_frames_benchmark = self.config.num_frames_benchmark()
        sample_params.lod = [[1.0] * self.config.num_levels()] * self.config.num_frames_recorder()
        sample_params.res = [[1.0] * self.config.num_levels()] * self.config.num_frames_recorder()
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
            if not self.distance_lod:
                lod = []
                for i in range(self.config.num_levels()):
                    index = i % self.num_optim_levels
                    lod.append(kwargs["lod"+str(index)])
                sample_params.lod = [lod] * self.config.num_frames_recorder()
            else:
                lod_params = []
                for i in range(self.config.num_levels()):
                    index = i % self.num_optim_levels
                    lod_params.append(vkgs_py.vec4(kwargs["min_lod"+str(index)],
                                                   kwargs["max_lod"+str(index)],
                                                   kwargs["min_dist"+str(index)],
                                                   kwargs["max_dist"+str(index)]))
                sample_params.lod_params = [lod_params] * self.config.num_frames_recorder()

            if self.config.dynamic_res():
                res = []
                for i in range(self.config.num_levels()):
                    index = i % self.num_optim_levels
                    res.append(kwargs["res"+str(index)])
                sample_params.res = [res] * self.config.num_frames_recorder()
            else:
                sample_params.res = [[1.0] * self.config.num_levels()] * self.config.num_frames_recorder()

            # add hard constraint
            if self.distance_lod:
                for i in range(self.num_optim_levels):
                    min_lod = kwargs["min_lod" + str(i)]
                    max_lod = kwargs["max_lod" + str(i)]
                    if min_lod > max_lod:
                        print(str(iteration) + "\t" + "skipped...")
                        iteration += 1
                        return 0

            sample_result = self.engine.sample(sample_params, sample_state)
            data_pred = vkgs_py.get_sample_result_numpy(sample_result)
        
            fixation_point = torch.tensor([sample_state.center[0].x * self.config.res()[0],
                                           sample_state.center[0].y * self.config.res()[1]])

            value, stats = self.fv.predict(
                data_pred[0],
                data_gt[0],
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
                if not self.distance_lod:
                    for i in range(self.config.num_levels()):
                        index = i % self.num_optim_levels
                        out_string += str(round(lod[index], 3)) + "\t"
                else:
                    for i in range(self.config.num_levels()):
                        index = i % self.num_optim_levels
                        out_string += str(round(lod_params[index].x, 3)) + "\t"
                        out_string += str(round(lod_params[index].y, 3)) + "\t"
                        out_string += str(round(lod_params[index].z, 3)) + "\t"
                        out_string += str(round(lod_params[index].w, 3)) + "\t"

                if self.config.dynamic_res():
                    for i in range(self.config.num_levels()):
                        index = i % self.num_optim_levels
                        out_string += str(round(res[index], 3)) + "\t"

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
            if not self.distance_lod:
                for i in range(self.config.num_levels()):
                    index = i % self.num_optim_levels                    
                    out_string += "*lod" + str(index) + "\t"
            else:
                for i in range(self.config.num_levels()):
                    index = i % self.num_optim_levels
                    out_string += "*minl" + str(index) + "\t"
                    out_string += "*maxl" + str(index) + "\t"
                    out_string += "*mind" + str(index) + "\t"
                    out_string += "*maxd" + str(index) + "\t"
    
            if self.config.dynamic_res():
                for i in range(self.config.num_levels()):
                    index = i % self.num_optim_levels
                    out_string += "*res" + str(index) + "\t"

            out_string += "qual\t"
            out_string += "time\t"
            out_string += "speedup\t"
            out_string += "loss"
            print(out_string)

        optimizer.maximize(
            init_points=10,
            n_iter=30,
        )

        max_lod_keys = sorted([x for x in optimizer.max["params"] if x.startswith("lod")])
        max_lod = [optimizer.max["params"][x] for x in max_lod_keys]
        max_res_keys = sorted([x for x in optimizer.max["params"] if x.startswith("res")])
        max_res = [optimizer.max["params"][x] for x in max_res_keys]
        max_lod_params_keys = sorted([x for x in optimizer.max["params"] if x.startswith("lod_params")])
        max_lod_params = [optimizer.max["lod_params"][x] for x in max_lod_params_keys]
        print("Target:", optimizer.max["target"], "\tLOD:", max_lod)
        print("Target:", optimizer.max["target"], "\tRes:", max_res)
        print("Target:", optimizer.max["target"], "\tLOD Params", max_lod_params)

        output = {}
        output["target"] = optimizer.max["target"]
        output["params"] = vkgs_py.SampleParams()
        output["params"].num_frames_recorder = self.config.num_frames_recorder()
        output["params"].num_frames_benchmark = self.config.num_frames_benchmark()
        
        if not self.distance_lod:
            output["params"].lod = [max_lod] * self.config.num_frames_recorder()
            output["params"].lod_params = [[vkgs_py.vec4(*self.default_lod_params)] * self.config.num_levels()] * self.config.num_frames_recorder()
        else:
            output["params"].lod_params = [max_lod_params] * self.config.num_frames_recorder()
            output["params"].lod = [[1.0] * self.config.num_levels()] * self.config.num_frames_recorder()

        if self.config.dynamic_res():
            output["params"].res = [max_res] * self.config.num_frames_recorder()
        else:
            output["params"].res = [[1.0] * self.config.num_levels()] * self.config.num_frames_recorder()

        # visualize
        if "samples" in self.visualize:
            visualize_optimizer(optimizer)

        if "render" in self.visualize:
            sample_result = self.engine.sample(output["params"], sample_state)
            data_pred = vkgs_py.get_sample_result_numpy(sample_result)

            plt.clf()
            plt.imshow(data_pred[0])
            plt.show()

        if "gaze_heatmap" in self.visualize:
            sample_result = self.engine_gt.sample(output["params"], sample_state)
            data_gt = vkgs_py.get_sample_result_numpy(sample_result)

            center = np.array([sample_state.center[0].x, sample_state.center[0].y])
            self.gaze_heatmap_visualizer.insert(center, output["params"])
            self.gaze_heatmap_visualizer.visualize(data_gt)

        self.iterations += 1
        return output