# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import itertools
import os
import random
import sys
from typing import List

from bayes_opt import BayesianOptimization, UtilityFunction
from colorama import Fore
from dataclasses import dataclass
import joblib
import numpy as np
import scipy.stats.qmc as qmc
from scipy.spatial.transform import Rotation
from PIL import Image
import torch
from tqdm import tqdm

from colmap import convert_colmap_camera, get_gaze_dir_global
from dataset import list_of_quat_to_torch, list_of_vec3_to_torch
from sampling import stratified_random

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


@dataclass
class SceneSamplerParams:
    """
    Sampler parameters and constraints
    """
    min_point: torch.Tensor = torch.zeros((3))  # min point constraint
    max_point: torch.Tensor = torch.zeros((3))  # max point constraint
    num_frames: int = 0


class SceneSampler(ABC):
    """
    Scene sampler
    """
    def __init__(
        self,
        sampler_params,
        num_gaze_dir,
        visualize: List[str]=[],
    ):
        self.sampler_params = sampler_params
        self.num_gaze_dir = num_gaze_dir
        self.visualize = visualize

    @abstractmethod
    def find_new_sample(self):
        pass
    
    def find_new_gaze_dir(self):
        return


class SceneSamplerActiveLearning(SceneSampler):
    """
    Scene sampler with active learning
    """
    def __init__(
        self,
        config,
        engine,
        time_budget: float,
        sample_state_template: vkgs_py.SampleState,  # example SampleState from COLMAP
        sampler_params,
        verbose: bool=False,
        num_gaze_dir: int=0,
        distance_lod: bool=False,
        single_level: bool=False,
        default_lod_params: List[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        super().__init__(sampler_params, num_gaze_dir)
        
        self.config = config
        self.engine = engine
        self.time_budget = time_budget
        self.sample_state_template = sample_state_template
        self.verbose = verbose
        self.distance_lod = distance_lod
        self.single_level = single_level
        self.default_lod_params = default_lod_params
        
        min_point = sampler_params.min_point
        max_point = sampler_params.max_point
        self.pbounds = {
            "p_x": (min_point[0], max_point[0]),
            "p_y": (min_point[1], max_point[1]),
            "p_z": (min_point[2], max_point[2]),
            "r_x": (-np.pi, np.pi),
            "r_y": (-np.pi, np.pi),
            "r_z": (-np.pi, np.pi),
        }

    
    def find_new_sample(
        self,
        dataset,
        model: torch.nn.Module,
        device: torch.device,
        sequence_length: int,
        num_samples: int,
    ):
        model.eval()

        # only works with models that output LOD parameters
        num_frames = self.sampler_params.num_frames
        
        sample_state = vkgs_py.SampleState()
        sample_state.pos = self.sample_state_template.pos
        sample_state.quat = self.sample_state_template.quat
        sample_state.center = self.sample_state_template.center
        sample_state.gaze_dir = self.sample_state_template.gaze_dir
        sample_state.view_angles = self.sample_state_template.view_angles

        sample_params = vkgs_py.SampleParams()
        sample_params.num_frames_recorder = 0
        sample_params.num_frames_benchmark = sequence_length
        sample_params.res = [[1.0] * self.config.num_levels()] * sequence_length
        sample_params.lod_params = [[vkgs_py.vec4(*self.default_lod_params)] * self.config.num_levels()] * sequence_length
        
        max_target = 0
        iteration = 0

        # set up optimizer
        def loss_function(
            **kwargs
        ):
            nonlocal max_target
            nonlocal iteration
            
            sample_state.pos = [vkgs_py.vec3(kwargs["p_x"], kwargs["p_y"], kwargs["p_z"])] * sequence_length
            quat = Rotation.from_euler("XYZ", [kwargs["r_x"], kwargs["r_y"], kwargs["r_z"]]).as_quat()
            sample_state.quat = [vkgs_py.quat(quat[3], quat[0], quat[1], quat[2])] * sequence_length
            sample_state.center = [sample_state.center[0]] * sequence_length
            sample_state.view_angles = [sample_state.view_angles[0]] * sequence_length
            sample_state.gaze_dir = [sample_state.gaze_dir[0]] * sequence_length
            
            torch_sample_state = {
                "pos": list_of_vec3_to_torch(sample_state.pos),
                "quat": list_of_quat_to_torch(sample_state.quat),
            }

            device = torch.device("cpu")
            input_data = dataset.flatten_sample_state(torch_sample_state).to(device)
            model.to(device)
            with torch.no_grad():
                model_output = model(input_data)

            sample_params.lod = model_output.tolist()
            result = self.engine.sample(sample_params, sample_state)
            render_time = np.median(np.array(result.time))

            target = abs(render_time - self.time_budget)

            if self.verbose:
                out_string = str(iteration) + "\t"
                
                # position
                out_string += str(round(kwargs["p_x"], 3)) + "\t"
                out_string += str(round(kwargs["p_y"], 3)) + "\t"
                out_string += str(round(kwargs["p_z"], 3)) + "\t"

                # rotation
                out_string += str(round(kwargs["r_x"], 3)) + "\t"
                out_string += str(round(kwargs["r_y"], 3)) + "\t"
                out_string += str(round(kwargs["r_z"], 3)) + "\t"
                
                out_string += str(round(render_time, 3)) + "\t"
                out_string += str(round(target, 3)) + "\t"

                if target > max_target:
                    max_target = target
                    print(Fore.GREEN + out_string + Fore.WHITE)
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
                
            # position
            out_string += "p_x\t"
            out_string += "p_y\t"
            out_string += "p_z\t"

            # rotation
            out_string += "r_x\t"
            out_string += "r_y\t"
            out_string += "r_z\t"

            out_string += "time\t"
            out_string += "loss\t"

            print(out_string)
        
        optimizer.maximize(
            init_points=num_samples,
            n_iter=num_samples,
        )

        model.train()
        
        max_pos_keys = sorted([x for x in optimizer.max["params"] if x.startswith("p_")])
        max_pos = [optimizer.max["params"][x] for x in max_pos_keys]
        max_rot_keys = sorted([x for x in optimizer.max["params"] if x.startswith("r_")])
        max_rot = [optimizer.max["params"][x] for x in max_rot_keys]
        print("Target:", optimizer.max["target"], "\tPos:", max_pos)
        print("Target:", optimizer.max["target"], "\tRot:", max_rot)
        
        sample_state.pos = [vkgs_py.vec3(max_pos[0], max_pos[1], max_pos[2])]
        quat = Rotation.from_euler("XYZ", [max_rot[0], max_rot[1], max_rot[2]]).as_quat()
        sample_state.quat = [vkgs_py.quat(quat[3], quat[0], quat[1], quat[2])]
        sample_state.center = [sample_state.center[0]]
        sample_state.gaze_dir = [sample_state.gaze_dir[0]]
        sample_state.view_angles = [sample_state.view_angles[0]]

        return sample_state
            

class SceneSamplerStratisfied(SceneSampler):
    """
    Scene sampler with stratisfied sampling
    """
    def __init__(
        self,
        config,
        engine,
        time_budget: float,
        sample_state_template: vkgs_py.SampleState,  # example SampleState from COLMAP
        sampler_params,
        data_dir,
        verbose: bool=False,
        num_gaze_dir: int=0,
        distance_lod: bool=False,
        single_level: bool=False,
        default_lod_params: List[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        super().__init__(sampler_params, num_gaze_dir)

        self.config = config
        self.engine = engine
        self.time_budget = time_budget
        self.sample_state_template = sample_state_template
        self.verbose = verbose
        self.distance_lod = distance_lod
        self.single_level = single_level
        self.default_lod_params = default_lod_params

        self.min_point = sampler_params.min_point
        self.max_point = sampler_params.max_point
        self.bins = [16, 16, 16]

        self.colmap_data = joblib.load(os.path.join(data_dir, "train.pkl"))


    def find_new_sample(
        self,
        dataset_train,
        dataset_time,
        sequence_length: int,
    ):
        # only works with models that output LOD parameters
        num_frames = self.sampler_params.num_frames
                
        positions = stratified_random(np.stack((self.min_point, self.max_point), axis=1), self.bins)        
        
        z_near = 0.01
        z_far = 100.0

        model_transform = vkgs_py.mat4_to_numpy(self.engine.get_model_matrix())
        
        progress = tqdm(total=np.array(positions.shape[:3]).prod().item())
        for prod in itertools.product(range(self.bins[0]), range(self.bins[1]), range(self.bins[2])):
            pos = positions[prod[0], prod[1], prod[2]]
            
            # populate state
            sample_state = vkgs_py.SampleState()
            sample_state.center = self.sample_state_template.center
            sample_state.gaze_dir = self.sample_state_template.gaze_dir
            sample_state.view_angles = self.sample_state_template.view_angles

            sample_state.pos = [vkgs_py.vec3(pos[0], pos[1], pos[2])] * sequence_length

            camera_index = np.random.randint(len(dataset_train))
            original_data = dataset_train.get_original_data(camera_index)

            #sample_state.quat = original_data["sample_state"].quat * sequence_length
            sample_state.center = [sample_state.center[0]] * sequence_length
            sample_state.view_angles = [sample_state.view_angles[0]] * sequence_length

            view_params_key = list(self.colmap_data.keys())[camera_index]
            view_params = self.colmap_data[view_params_key]
            camera_data = convert_colmap_camera(view_params, model_transform, z_near, z_far)
            gaze_dir = get_gaze_dir_global(camera_data["cam_mat"], camera_data["proj_mat"], np.array([sample_state.center[0].x, sample_state.center[0].y]))
            sample_state.gaze_dir = [vkgs_py.vec3(gaze_dir[0], gaze_dir[1], gaze_dir[2])] * sequence_length
            sample_state.quat = [camera_data["sample_state"].quat[0]] * sequence_length

            # populate parameters
            sample_params = vkgs_py.SampleParams()
            sample_params.num_frames_recorder = 0
            sample_params.num_frames_benchmark = sequence_length
            sample_params.res = [[1.0] * self.config.num_levels()] * sequence_length
            sample_params.lod = [[1.0] * self.config.num_levels()] * sequence_length
            sample_params.lod_params = [[vkgs_py.vec4(*self.default_lod_params)] * self.config.num_levels()] * sequence_length

            random_lod = random.uniform(0.0, 1.0)
            sample_params.lod = [[random_lod] * self.config.num_levels()] * sequence_length

            results = self.engine.sample(sample_params, sample_state)

            metadata = {}
            metadata["mode"] = "stratified"
            metadata["res"] = self.config.res()

            result_data = {}
            result_data["time"] = np.median(results.time).item()
            
            dataset_time.insert(
                sample_state,
                sample_params,
                result_data=result_data,
                metadata=metadata)
            
            progress.update(1)

        progress.close()


class SceneSamplerCOLMAP(SceneSampler):
    """
    Scene sampler with image
    """
    def __init__(
        self,
        sampler_params,
        num_gaze_dir,
        image_dir,
        data_dir,
        engine,
        visualize: List[str]=[],
    ):
        super().__init__(sampler_params, num_gaze_dir)
        
        self.image_dir = image_dir
        self.engine = engine
        self.view_count = 0
        self.gaze_dir_count = 0
        self.sampler = qmc.Halton(2, scramble=False)

        self.colmap_data = joblib.load(os.path.join(data_dir, "train.pkl"))


    def get_sample_state(self, index, center=vkgs_py.vec2(0.5, 0.5)):
        z_near = 0.01
        z_far = 100.0

        model_transform = vkgs_py.mat4_to_numpy(self.engine.get_model_matrix())
        view_params_key = list(self.colmap_data.keys())[index]
        view_params = self.colmap_data[view_params_key]
        camera_data = convert_colmap_camera(view_params, model_transform, z_near, z_far)
        sample_state = camera_data["sample_state"]
        
        gaze_dir = get_gaze_dir_global(camera_data["cam_mat"], camera_data["proj_mat"], np.array([center.x, center.y]))
        gaze_dir = vkgs_py.vec3(gaze_dir[0], gaze_dir[1], gaze_dir[2])

        sample_state.pos = sample_state.pos * self.sampler_params.num_frames
        sample_state.quat = sample_state.quat * self.sampler_params.num_frames
        sample_state.view_angles = sample_state.view_angles * self.sampler_params.num_frames
        
        # compute other sample state attributes
        sample_state.center = [center] * self.sampler_params.num_frames
        sample_state.gaze_dir = [gaze_dir] * self.sampler_params.num_frames
    
        return sample_state


    def find_new_sample(self):
        if self.num_gaze_dir > 0:
            center = self.sampler.random(1)[0]
            center = vkgs_py.vec2(center[0], center[1])

        sample_state = self.get_sample_state(self.view_count)

        self.gaze_dir_count += 1
        if self.gaze_dir_count >= self.num_gaze_dir:
            self.gaze_dir_count = 0
            self.view_count += 1

        return sample_state


    def get_num_samples(self):
        if self.num_gaze_dir == 0:
            return len(self.colmap_data.keys())
        return len(self.colmap_data.keys()) * self.num_gaze_dir


    def get_pos_bounds(self):
        pos = []
        for i in range(self.get_num_samples()):
            sample_state = self.get_sample_state(i)
            pos.append([sample_state.pos[0].x, sample_state.pos[0].y, sample_state.pos[0].z])
        pos = np.array(pos)

        output = {}
        output["min"] = np.min(pos, axis=0)
        output["max"] = np.max(pos, axis=0)
        return output
    