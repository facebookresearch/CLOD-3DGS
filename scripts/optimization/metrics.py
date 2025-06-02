# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import random

import numpy as np
import pyfvvdp


def get_pyfvvdp_metric(width, height, mode, foveated, device, heatmap=None):
    if mode not in ["standard_fhd"]:
        raise ValueError("mode is not valid")

    # based on "standard_fhd" configuration
    display_geometry = pyfvvdp.fvvdp_display_geometry(
        resolution = [width, height],
        distance_m = 0.6,
        diagonal_size_inches = 24,
    )
    display_photometry = pyfvvdp.fvvdp_display_photo_eotf(
        contrast = 1000,
        E_ambient = 250,
        Y_peak = 200,
    )

    return pyfvvdp.fvvdp(
        display_geometry=display_geometry,
        display_photometry=display_photometry,
        device=device,
        foveated=foveated,
        heatmap=heatmap
    )


def convert_eccentricity_to_radius(eccentricity: float, width: int, height: int):
    """
    Args:
        width: screen width (pixels)
        height: screen height (pixels)

    Returns:
        radius: ratio between of the height of the display [0.0, 1.0]
    """
    distance_m = 0.6
    distance_inches = 39.37 * distance_m
    diagonal_size_inches = 24
    
    eccentricity_radians = np.radians(eccentricity)

    theta = np.arctan(height / width)
    display_width = np.cos(theta) * diagonal_size_inches
    display_height = np.sin(theta) * diagonal_size_inches

    fov_x = np.arctan((display_width / 2) / distance_inches) * 2
    fov_y = np.arctan((display_height / 2) / distance_inches) * 2

    radius_height = np.tan(eccentricity_radians) * (display_height / 2)

    return (radius_height / (display_height / 2)).item()
