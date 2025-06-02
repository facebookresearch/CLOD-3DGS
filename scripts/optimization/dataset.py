# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from typing import Dict

import orjson
import torch
from torch.utils.data import Dataset
import yaml

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


def list_of_quat_to_torch(quat_list):
    output = torch.zeros((len(quat_list), 4), dtype=torch.float32)
    for i in range(len(quat_list)):
        output[i, 0] = quat_list[i].w
        output[i, 1] = quat_list[i].x
        output[i, 2] = quat_list[i].y
        output[i, 3] = quat_list[i].z
    return output


def list_of_vec2_to_torch(vec2_list):
    output = torch.zeros((len(vec2_list), 2), dtype=torch.float32)
    for i in range(len(vec2_list)):
        output[i, 0] = vec2_list[i].x
        output[i, 1] = vec2_list[i].y
    return output


def list_of_vec3_to_torch(vec3_list):
    output = torch.zeros((len(vec3_list), 3), dtype=torch.float32)
    for i in range(len(vec3_list)):
        output[i, 0] = vec3_list[i].x
        output[i, 1] = vec3_list[i].y
        output[i, 2] = vec3_list[i].z
    return output


def list_of_vec4_to_torch(vec4_list):
    output = torch.zeros((len(vec4_list), 4), dtype=torch.float32)
    for i in range(len(vec4_list)):
        output[i, 0] = vec4_list[i].x
        output[i, 1] = vec4_list[i].y
        output[i, 2] = vec4_list[i].z
        output[i, 3] = vec4_list[i].w
    return output


def list_of_view_frustum_angles_to_torch(view_frustum_angles_list):
    output = torch.zeros((len(view_frustum_angles_list), 4), dtype=torch.float32)
    for i in range(len(view_frustum_angles_list)):
        output[i, 0] = view_frustum_angles_list[i].angle_right
        output[i, 1] = view_frustum_angles_list[i].angle_left
        output[i, 2] = view_frustum_angles_list[i].angle_down
        output[i, 3] = view_frustum_angles_list[i].angle_up
    return output


class SceneParamsDataset(Dataset):
    def __init__(
        self,
        split,
        input_features,
        num_frames,
        num_levels,
    ):
        self.split = split
        self.input_features = input_features
        self.num_frames = num_frames
        self.num_levels = num_levels

        self.original_data = []
        self.torch_data = []
    
    
    def load_data(
        self,
        filename,
    ):
        self.original_data.clear()
        self.torch_data.clear()

        with open(filename) as f:
            if filename.endswith(".yaml"):
                loaded_data = yaml.safe_load(f)
            elif filename.endswith(".json"):
                loaded_data = orjson.loads(f.read())
            
            temp_keys = sorted(loaded_data.keys())
            keys = []
            index = 0
            for key in temp_keys:
                if self.split == "train" and index % 8 != 0:
                    keys.append(key)
                elif self.split == "valid" and index % 8 == 0:
                    keys.append(key)
                elif self.split == "all":
                    keys.append(key)
                index += 1

            for key in keys:
                sample = loaded_data[key]

                # sample state
                sample_state = vkgs_py.SampleState()
                sample_state.pos = [vkgs_py.vec3(*pos) for pos in sample["sample_state"]["pos"]]
                sample_state.quat = [vkgs_py.quat(*quat) for quat in sample["sample_state"]["quat"]]
                sample_state.center = [vkgs_py.vec2(*center) for center in sample["sample_state"]["center"]]
                sample_state.gaze_dir = [vkgs_py.vec3(*gaze_dir) for gaze_dir in sample["sample_state"]["gaze_dir"]]
                sample_state.view_angles = [vkgs_py.ViewFrustumAngles(*gaze_dir) for gaze_dir in sample["sample_state"]["view_angles"]]
                
                # sample state
                sample_params = vkgs_py.SampleParams()
                if "lod" in sample["sample_params"]:
                    sample_params.lod = sample["sample_params"]["lod"]
                if "res" in sample["sample_params"]:
                    sample_params.res = sample["sample_params"]["res"]
                if "lod_params" in sample["sample_params"]:
                    sample_params.lod_params = [[vkgs_py.vec4(*x) for x in lod_params] for lod_params in sample["sample_params"]["lod_params"]]

                metadata = {}
                for key, value in sample["metadata"].items():
                    metadata[key] = value

                result_data = {}
                if "result_data" in sample:
                    for key, value in sample["result_data"].items():
                        result_data[key] = value

                # sample state
                self.insert(
                    sample_state=sample_state,
                    sample_params=sample_params,
                    metadata=metadata,
                    result_data=result_data,
                )
        
        return loaded_data

    
    def insert(
        self,
        sample_state: vkgs_py.SampleState,
        sample_params: vkgs_py.SampleParams,
        result_data: Dict={},
        metadata: Dict={},
    ):
        sample = {
            "sample_state": {},
            "sample_params": {},
            "result_data": {},
            "metadata": {},
        }

        if "pos" in self.input_features:
            sample["sample_state"]["pos"] = list_of_vec3_to_torch(sample_state.pos)
        if "quat" in self.input_features:
            sample["sample_state"]["quat"] = list_of_quat_to_torch(sample_state.quat)
        if "gaze_dir" in self.input_features:
            sample["sample_state"]["gaze_dir"] = list_of_vec3_to_torch(sample_state.gaze_dir)
        if "center" in self.input_features:
            sample["sample_state"]["center"] = list_of_vec2_to_torch(sample_state.center)
        if "view_angles" in self.input_features:
            sample["sample_state"]["view_angles"] = list_of_view_frustum_angles_to_torch(sample_state.view_angles)

        sample["sample_params"]["lod"] = torch.tensor(sample_params.lod)
        if "target" in result_data:
            sample["result_data"]["target"] = torch.tensor(result_data["target"])
        if "time" in result_data:
            sample["result_data"]["time"] = torch.tensor(result_data["time"])

        if "res" in metadata:
            sample["metadata"]["res"] = metadata["res"]

        self.torch_data.append(sample)
        self.original_data.append({
            "sample_state": sample_state,
            "sample_params": sample_params,
            "result_data": result_data,
            "metadata": metadata,
        })


    def __len__(self):
        return len(self.torch_data)
        

    def __getitem__(self, idx):
        return self.torch_data[idx]
    

    def get_original_data(self, idx):
        return self.original_data[idx]
    

    def save(self, filename):
        output = {}
        
        index = 0
        for data_sample in self.original_data:
            output[index] = {"sample_state": {}, "sample_params": {}, "metadata": {}, "result_data": {}}

            output[index]["sample_state"]["pos"] = [[pos.x, pos.y, pos.z] for pos in data_sample["sample_state"].pos]
            output[index]["sample_state"]["quat"] = [[quat.w, quat.x, quat.y, quat.z] for quat in data_sample["sample_state"].quat]
            output[index]["sample_state"]["view_angles"] = [[v.angle_right, v.angle_left, v.angle_down, v.angle_up] for v in data_sample["sample_state"].view_angles]
            output[index]["sample_state"]["gaze_dir"] = [[gaze_dir.x, gaze_dir.y, gaze_dir.z] for gaze_dir in data_sample["sample_state"].gaze_dir]
            output[index]["sample_state"]["center"] = [[center.x, center.y] for center in data_sample["sample_state"].center]

            output[index]["sample_params"]["lod"] = data_sample["sample_params"].lod

            for key in data_sample["metadata"].keys():
                output[index]["metadata"][key] = data_sample["metadata"][key]

            for key in data_sample["result_data"].keys():
                output[index]["result_data"][key] = data_sample["result_data"][key]

            index += 1

        with open(filename, "w") as f:
            if filename.endswith(".yaml"):
                yaml.dump(output, f, default_flow_style=False)
            elif filename.endswith(".json"):
                json.dump(output, f)
    

    def flatten_sample_state(self, sample_state, num_frames=None):
        if not num_frames:
            num_frames = self.num_frames

        data = []
        for key in sample_state.keys():
            data.append(torch.flatten(sample_state[key][:, :num_frames], start_dim=1))
        data = torch.cat(data, dim=-1)
        return data


    def flatten_sample_params(self, sample_params, num_frames=1):
        if not num_frames:
            num_frames = self.num_frames

        data = []
        for key in sample_params.keys():
            data.append(torch.flatten(sample_params[key][:, :num_frames], start_dim=1))
        data = torch.cat(data, dim=-1)
        return data


    def get_num_inputs(self, num_frames=None):
        if not num_frames:
            num_frames = self.num_frames

        num_inputs = 0
        if "pos" in self.input_features:
            num_inputs += 3 * num_frames
        if "quat" in self.input_features:
            num_inputs += 4 * num_frames
        if "gaze_dir" in self.input_features:
            num_inputs += 3 * num_frames
        if "center" in self.input_features:
            num_inputs += 2 * num_frames
        if "view_angles" in self.input_features:
            num_inputs += 4 * num_frames
        if "time" in self.input_features:
            num_inputs += 1
        return num_inputs


    def get_num_outputs(self):
        num_outputs = self.num_levels  # for LOD
        return num_outputs
