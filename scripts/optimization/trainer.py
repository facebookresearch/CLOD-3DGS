# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import random
import sys
from typing import List

import numpy as np
import pyfvvdp
import torch
from torch.utils.data import DataLoader
import yaml

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py

from dataset import SceneParamsDataset
from model import SceneParamsModel
from scene_sampler import SceneSamplerStratisfied, SceneSamplerActiveLearning, SceneSamplerCOLMAP, SceneSamplerParams
from view_sampler import ViewSampler


class Trainer:
    def __init__(
        self,
        filename_input: str,
        filename_gt: str,
        config_input: vkgs_py.Config,
        config_gt: vkgs_py.Config,
        time_budget: float,
        verbose: bool=False,
        debug: bool=False,
        visualize: List[str]=[],
        image_dir: str=None,
        data_dir: str=None,
        model_name: str=None,
        num_gaze_dir: int=0,
        distance_lod: bool=False,
        single_level: bool=False,
        default_lod_params: List[float]=[1.0, 1.0, 1.0, 1.0],
    ):
        self.config = config_input
        self.time_budget = time_budget

        self.image_dir = image_dir
        self.data_dir = data_dir
        self.model_name = model_name
        self.batch_size = 128

        input_features = ["pos", "quat", "time"]
        
        self.iterations = 0
        self.dataset_train = SceneParamsDataset(
            split="train",
            input_features=input_features,
            num_frames=config_input.num_frames_recorder(),
            num_levels=config_input.num_levels())
        self.dataset_val = SceneParamsDataset(
            split="valid",
            input_features=input_features,
            num_frames=config_input.num_frames_recorder(),
            num_levels=config_input.num_levels())

        self.dataset_time = SceneParamsDataset(
            split="all",
            input_features=input_features,
            num_frames=config_input.num_frames_recorder(),
            num_levels=config_input.num_levels())

        # scene sampler
        scene_sampler_params = SceneSamplerParams()
        scene_sampler_params.min_point = [-1.0, 0.5, -1.0]
        scene_sampler_params.max_point = [1.0, 0.5, 1.0]
        scene_sampler_params.num_frames = self.config.num_frames_recorder()

        # parameterized engine
        self.engine = vkgs_py.Engine(self.config)
        self.engine.load_splats(filename_input)
        self.engine.start()

        self.scene_sampler_colmap = SceneSamplerCOLMAP(
            sampler_params=scene_sampler_params,
            num_gaze_dir=num_gaze_dir,
            image_dir=image_dir,
            data_dir=data_dir,
            engine=self.engine,
        )

        pos_bounds = self.scene_sampler_colmap.get_pos_bounds()

        scene_sampler_params.min_point = pos_bounds["min"].tolist()
        scene_sampler_params.max_point = pos_bounds["max"].tolist()

        self.scene_sampler_active_learning = SceneSamplerActiveLearning(
            config=self.config,
            engine=self.engine,
            time_budget=self.time_budget,
            sample_state_template=self.scene_sampler_colmap.get_sample_state(0),
            sampler_params=scene_sampler_params,
            verbose=verbose,
            num_gaze_dir=num_gaze_dir,
            distance_lod=distance_lod,
            single_level=single_level,
            default_lod_params=default_lod_params,
        )

        self.scene_sampler_stratisfied = SceneSamplerStratisfied(
            config=self.config,
            engine=self.engine,
            time_budget=self.time_budget,
            sample_state_template=self.scene_sampler_colmap.get_sample_state(0),
            sampler_params=scene_sampler_params,
            verbose=verbose,
            data_dir=data_dir,
            num_gaze_dir=num_gaze_dir,
            distance_lod=distance_lod,
            single_level=single_level,
            default_lod_params=default_lod_params,
        )

        # setup model and optimizer
        self.device = torch.device("cuda:0")
        #self.device = torch.device("cpu")
        self.model = SceneParamsModel(
            in_channels=self.dataset_train.get_num_inputs(num_frames=1),
            out_channels=self.dataset_train.get_num_outputs(),
            hidden_channels=32,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.MSELoss()

        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        self.best_validation_loss = float("inf")

        # ground truth engine
        self.config_gt = config_gt
        self.engine_gt = vkgs_py.Engine(self.config_gt)
        self.engine_gt.load_splats(filename_gt)
        self.engine_gt.start()
        
        # view sampler
        self.view_sampler = ViewSampler(
            config=self.config,
            engine=self.engine,
            engine_gt=self.engine_gt,
            time_budget=self.time_budget,
            verbose=verbose,
            debug=debug,
            visualize=visualize,
            num_gaze_dir=num_gaze_dir,
            distance_lod=distance_lod,
            single_level=single_level,
            default_lod_params=default_lod_params,
        )


    def get_sample_colmap(self):
        sample_state = self.scene_sampler_colmap.find_new_sample()
        return sample_state


    def get_sample_active_learning(self, sequence_length: int, num_samples: int):
        sample_state = self.scene_sampler_active_learning.find_new_sample(self.dataset_train, self.model, self.device, sequence_length, num_samples)
        return sample_state


    def get_sample_stratisfied(self, sequence_length: int):
        self.scene_sampler_stratisfied.find_new_sample(self.dataset_train, self.dataset_time, sequence_length)


    def get_num_samples_colmap(self):
        return self.scene_sampler_colmap.get_num_samples()


    def sample_iteration(self, sample_state, mode):
        sample = self.view_sampler.train(sample_state=sample_state)
        results_data = {}
        results_data["target"] = sample["target"]

        metadata = {}
        metadata["mode"] = mode
        metadata["res"] = self.config.res()
        
        self.dataset_train.insert(
            sample_state,
            sample["params"],
            metadata=metadata,
        )

        if self.data_dir is not None:
            self.dataset_train.save(os.path.join(self.data_dir, "models", self.model_name, "train.yaml"))

        self.iterations += 1


    def save_model(self, name: str):
        torch.save(self.best_state_dict, os.path.join(self.data_dir, "models", self.model_name, name + ".pth"))
        self.save_model_config(os.path.join(self.data_dir, "models", self.model_name, name + ".yaml"))
        np.savez(os.path.join(self.data_dir, "models", self.model_name, name + ".npz"), **self.model.export_to_numpy())


    def load_model(self, name:str):
        self.model.load_state_dict(torch.load(os.path.join(self.data_dir, "models", self.model_name, name + ".pth")))


    def save_model_config(self, filename, num_frames=1):
        config = {"sample_state": {}, "sample_params": {}, "metadata": {}, "result_data": {}}

        sample_torch = self.dataset_train[0]
        sample_original = self.dataset_train.original_data[0]

        for key, value in sample_torch["sample_state"].items():
            config["sample_state"][key] = {"shape": list(value[[0]].shape)}

        if "time" in self.dataset_train.input_features:
            config["result_data"]["time"] = {"shape": [1, 1]}

        for key, value in sample_torch["sample_params"].items():
            config["sample_params"][key] = {"shape": list(value[[0]].shape)}

        config["metadata"]["res"] = sample_original["metadata"]["res"]

        with open(filename, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


    def train_epoch(self):
        self.model.train()
        dataloader_train = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

        losses_epoch = []
        for batch, data in enumerate(dataloader_train):
            input_data = self.dataset_train.flatten_sample_state(data["sample_state"], num_frames=1).to(self.device)

            if "time" in self.dataset_train.input_features:
                input_data = torch.cat((input_data, data["result_data"]["time"][:, None].to(self.device)), dim=-1)

            gt_data = self.dataset_train.flatten_sample_params(data["sample_params"], num_frames=1).to(self.device)
            
            pred_data = self.model(input_data)
            
            loss = self.loss_fn(pred_data, gt_data)
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            losses_epoch.append(loss.item())

        output = {}
        output["loss"] = np.array(losses_epoch).mean().item()
        return output
    
    
    def validate_epoch(self):
        self.model.eval()
        dataloader_val = DataLoader(self.dataset_val, batch_size=1, shuffle=False)

        losses_epoch = []
        with torch.no_grad():
            for batch, data in enumerate(dataloader_val):
                input_data = self.dataset_val.flatten_sample_state(data["sample_state"], num_frames=1).to(self.device)

                if "time" in self.dataset_val.input_features:
                    input_data = torch.cat((input_data, data["result_data"]["time"][:, None].to(self.device)), dim=-1)

                gt_data = self.dataset_val.flatten_sample_params(data["sample_params"], num_frames=1).to(self.device)

                pred_data = self.model(input_data)
            
                loss = self.loss_fn(pred_data, gt_data)

                losses_epoch.append(loss.item())

        output = {}
        output["loss"] = np.array(losses_epoch).mean().item()

        if self.best_validation_loss >= output["loss"]:
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_validation_loss = output["loss"]
            print("best loss", self.best_validation_loss)

        return output


    def shutdown(self):
        self.engine.end()
        self.engine_gt.end()
