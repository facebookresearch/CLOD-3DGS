# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SceneParamsModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        self.linear0 = nn.Linear(in_channels, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)


    def forward(self, x):
        x0 = torch.relu(self.linear0(x))
        x1 = torch.relu(self.linear1(x0))
        x2 = torch.sigmoid(self.linear2(x1))
        return x2
    

    def export_to_numpy(self):
        output = {}
        output["linear0_weight"] = self.linear0.weight.detach().cpu().numpy()
        output["linear0_bias"] = self.linear0.bias.detach().cpu().numpy()
        output["linear1_weight"] = self.linear1.weight.detach().cpu().numpy()
        output["linear1_bias"] = self.linear1.bias.detach().cpu().numpy()
        output["linear2_weight"] = self.linear2.weight.detach().cpu().numpy()
        output["linear2_bias"] = self.linear2.bias.detach().cpu().numpy()
        return output