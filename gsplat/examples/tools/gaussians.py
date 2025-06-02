# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from geometry import surface_area_ellipsoid


class Gaussians:
    def __init__(self, params):
        self.o_num_points = params["x"].shape[0]
        self.num_points = params["x"].shape[0]

        ########## geometry properties ##########
        self.xyz = np.stack(
            (params["x"], params["y"], params["z"]), axis=-1,
        )  # position
        self.n_xyz = None
        if "nx" in params:
            self.n_xyz = np.stack(
                (params["nx"], params["ny"], params["nz"]), axis=-1,
            )  # normal
        self.scale = np.stack(
            (params["scale_0"], params["scale_1"], params["scale_2"]), axis=-1,
        )  # scale
        self.rot = np.stack(
            (params["rot_0"], params["rot_1"], params["rot_2"], params["rot_3"]), axis=-1,
        )  # rotation
        self.opacity = params["opacity"][:, None]  # opacity

        ########## features ##########
        self.f_dc = np.stack(
            (params["f_dc_0"], params["f_dc_1"], params["f_dc_2"]), axis=-1,
        )  # f_dc

        num_features = len([x for x in params.keys() if x.startswith("f_rest_")])
        feature_arrays = [params["f_rest_" + str(x)] for x in range(num_features)]

        self.f_rest = np.stack(
            feature_arrays, axis=-1,
        )  # f_rest


    def subsample(self, indices):
        self.xyz = self.xyz[indices]
        if self.n_xyz is not None:
            self.n_xyz = self.n_xyz[indices]
        self.scale = self.scale[indices]
        self.rot = self.rot[indices]

        self.opacity = self.opacity[indices]

        self.f_dc = self.f_dc[indices]
        self.f_rest = self.f_rest[indices]

        self.num_points = len(indices)


    def subsample_basic(self, p):
        num_points_p = int(self.num_points * p)
        indices = np.sort(np.random.choice(self.num_points, size=num_points_p, replace=False))
        self.subsample(indices)


    def subsample_size(self, p):
        num_points_p = int(self.num_points * p)
        sa = surface_area_ellipsoid(
            np.exp(self.scale[..., 0]),
            np.exp(self.scale[..., 1]),
            np.exp(self.scale[..., 2]),
        )
        indices = np.argsort(sa)[::-1][:num_points_p]
        self.subsample(indices)


    def limit(self, limit):
        indices = list(range(0, limit))
        self.subsample(indices)


    def scale_gaussians(self, s):
        self.scale = np.log(np.exp(self.scale) * s)


    def get_params(self):
        output = {}

        output["x"] = self.xyz[..., 0]
        output["y"] = self.xyz[..., 1]
        output["z"] = self.xyz[..., 2]
        if self.n_xyz is not None:
            output["nx"] = self.n_xyz[..., 0]
            output["ny"] = self.n_xyz[..., 1]
            output["nz"] = self.n_xyz[..., 2]

        output["f_dc_0"] = self.f_dc[..., 0]
        output["f_dc_1"] = self.f_dc[..., 1]
        output["f_dc_2"] = self.f_dc[..., 2]

        for i in range(self.f_rest.shape[1]):
            output["f_rest_" + str(i)] = self.f_rest[..., i]

        output["opacity"] = self.opacity[...,]
        output["scale_0"] = self.scale[..., 0]
        output["scale_1"] = self.scale[..., 1]
        output["scale_2"] = self.scale[..., 2]
        output["rot_0"] = self.rot[..., 0]
        output["rot_1"] = self.rot[..., 1]
        output["rot_2"] = self.rot[..., 2]
        output["rot_3"] = self.rot[..., 3]

        return output
