# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np


def surface_area_ellipsoid(a, b, c):
    sa = ((a * b)**1.6) + ((a * c)**1.6) + ((b * c)**1.6)
    sa = (sa/3) ** (1.0/1.6)
    return 4 * np.pi * sa
