# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# COLMAP data processing script

import os
import sys
from typing import Tuple, Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

# vkgs_py
sys.path.insert(0, os.path.abspath("."))
import vkgs_py


def convert_colmap_camera(
    view_params: Dict,
    model_transform: np.ndarray,
    z_near: float,
    z_far: float
) -> Dict:
    """
    Convert COLMAP camera to OpenGL format

    Args:
        view_params: camera parameters
        model_transform: model offset transformation
        z_near: near culling plane
        z_far: far culling plane

    Returns:
        Dict: converted camera parameters
    """

    transform = np.matrix.transpose(model_transform)  # row major to column major
    sample_state = vkgs_py.SampleState()

    # extrinsics
    cam_mat = get_translation_matrix(view_params["pos"][0], view_params["pos"][1], view_params["pos"][2])
    cam_rot = R.from_quat(view_params["quat"]).as_matrix()  # [x, y, z, w]
    cam_rot[:, 1] *= -1  # convert from OpenCV to OpenGL
    cam_rot[:, 2] *= -1  # convert from OpenCV to OpenGL
    cam_mat[:3, :3] = cam_rot
    cam_mat = transform @ cam_mat
    
    sample_state.pos = [vkgs_py.vec3(
        cam_mat[0, 3],
        cam_mat[1, 3],
        cam_mat[2, 3]
    )]
    cam_rot = cam_mat[:3, :3]
    rot_quat = R.from_matrix(cam_rot[:3, :3]).as_quat().tolist()  # [x, y, z, w]
    sample_state.quat = [vkgs_py.quat(rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2])]  # [w, x, y, z]

    # intrinsics
    proj_mat = Ks_to_proj_mat(view_params["Ks"], view_params["image_size"], z_near, z_far)
    sample_state.view_angles = [get_view_angles(proj=proj_mat)]

    output = {}
    output["sample_state"] = sample_state
    output["proj_mat"] = proj_mat
    output["cam_mat"] = cam_mat
    return output


def get_translation_matrix(
    x: float,
    y: float,
    z: float
) -> np.ndarray:
    """
    Get translation matrix from (x, y, z) coordinates
    
    Args:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate

    Returns:
        np.ndarray: translation matrix
    """
    output = np.eye(4)
    output[0, 3] = x
    output[1, 3] = y
    output[2, 3] = z
    return output


def Ks_to_proj_mat(
    Ks: np.ndarray,  # [3, 3]
    image_size: Tuple[int],  # [2]
    z_near: float,
    z_far: float,
) -> np.ndarray:
    """
    Get translation matrix from (x, y, z) coordinates
    
    Args:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate

    Returns:
        np.ndarray: translation matrix
    """
    # adapted from https://stackoverflow.com/a/22064917
    fx = Ks[0, 0]
    fy = Ks[1, 1]
    s = Ks[0, 1]
    cx = Ks[0, 2]
    cy = Ks[1, 2]
    
    W = image_size[0]
    H = image_size[1]

    output = np.zeros((4, 4))
    output[0, 0] = 2 * fx / W
    output[1, 0] = 2 * s / W
    output[1, 1] = 2 * fy / H
    output[2, 0] = 2 * (cx / W) - 1
    output[2, 1] = 2 * (cy / H) - 1
    output[2, 2] = (z_far + z_near) / (z_far - z_near)
    output[2, 3] = 1
    output[3, 2] = 2 * z_far * z_near / (z_far - z_near)

    # convert to Vulkan
    output[1, 1] *= -1
    return output


def get_view_angle(vec: np.ndarray, inv_proj: np.ndarray) -> float:
    """
    Get view angle from inverse projection matrix and vector

    Args:
        vec: 3D vector
        inv_proj: inverse projection matrix [4, 4]

    Returns:
        float: angle
    """
    ray = vec @ inv_proj
    ray[2] = -1.0
    ray[3] = 0.0
    ray = ray / np.linalg.norm(ray[:3])
    ray_angle = np.arccos(np.dot(ray[:3], np.array((0.0, 0.0, -1.0))))
    return ray_angle


def get_view_angles(proj: np.ndarray) -> Dict[str, float]:
    """
    Gets view angles from projection matrix. This is a naive implementation where it just performs raycasting.

    Args:
        proj: projection matrix [4, 4]

    Returns:
        Dict: dictionary of view angles
    """
    inv_proj = np.linalg.inv(proj)
    view_angles = vkgs_py.ViewFrustumAngles()
    view_angles.angle_up = get_view_angle(vec=np.array((0.0, 1.0, 1.0, 1.0)), inv_proj=inv_proj)
    view_angles.angle_down = -get_view_angle(vec=np.array((0.0, -1.0, 1.0, 1.0)), inv_proj=inv_proj)
    view_angles.angle_right = get_view_angle(vec=np.array((1.0, 0.0, 1.0, 1.0)), inv_proj=inv_proj)
    view_angles.angle_left = -get_view_angle(vec=np.array((-1.0, 0.0, 1.0, 1.0)), inv_proj=inv_proj)
    return view_angles


def get_gaze_dir_local(proj: np.ndarray, pos_2d: np.ndarray) -> np.ndarray:
    """
    Get gaze direction in local camera space

    Args:
        proj: projection matrix [4, 4]
        pos_2d: 2D camera coordinates

    Returns:
        np.ndarray: 3D ray in camera space
    """
    inv_proj = np.linalg.inv(proj)
    pos = np.array([(2.0 * pos_2d[0]) - 1.0, (2.0 * pos_2d[1]) - 1.0, 1.0, 1.0])
    ray = pos @ inv_proj
    ray[2] = -1.0
    ray[3] = 0.0
    ray = ray / np.linalg.norm(ray[:3])
    return ray[:3]


def get_gaze_dir_global(cam: np.ndarray, proj: np.ndarray, pos_2d: np.ndarray) -> np.ndarray:
    """
    Get gaze direction in global camera space

    Args:
        cam: camera transformation matrix [4, 4]
        proj: projection matrix [4, 4]
        pos_2d: 2D camera coordinates

    Returns:
        np.ndarray: 3D ray in camera space
    """
    ray_local = get_gaze_dir_local(proj, pos_2d)
    cam_rot = cam[:3, :3]
    ray_global = cam_rot @ ray_local
    return ray_global


def extract_proj_parameters(proj: np.ndarray) -> Dict:
    """
    Extract projection parameters from projection matrix

    Args:
        proj: projection matrix [4, 4]

    Returns:
        Dict: dictionary of camera parameters
    """
    params = {}
    params["fovy"] = 2.0 * np.arctan(1.0 / proj[1][1])
    params["view_angles"] = get_view_angles(proj)
    params["cx"] = (1.0 - proj[0][2] * 0.5)
    params["cy"] = (1.0 + proj[1][2] * 0.5)
    return params


if __name__ == "__main__":
    proj = np.array([
        [0.487139, 0.0, 0.5, 0.0],
        [0.0, 0.866025, -0.5, 0.0],
        [0.0, 0.0, -1.0002, -0.020002],
        [0.0, 0.0, -1, 0.0],
    ])
    
    # test view angle extraction
    params = extract_proj_parameters(proj)
    assert(np.isclose(params["fovy"], 1.71414))
    assert(np.isclose(params["cx"], 0.75))
    assert(np.isclose(params["cy"], 0.75))
    assert(np.isclose(params["view_angles"].angle_up - params["view_angles"].angle_down, 1.71414))

    # test gaze 2D -> ray conversion
    ray_center = get_gaze_dir_local(proj, np.array([0.5, 0.5]))
    assert(ray_center[0] == 0.0)
    assert(ray_center[1] == 0.0)

    ray_right = get_gaze_dir_local(proj, np.array([1.0, 0.5]))
    ray_right_angle = np.arccos(np.dot(ray_right[:3], np.array((0.0, 0.0, -1.0))))
    assert(np.isclose(abs(params["view_angles"].angle_left), ray_right_angle))
    assert(ray_right[1] == 0.0)

    ray_left = get_gaze_dir_local(proj, np.array([0.0, 0.5]))
    ray_left_angle = np.arccos(np.dot(ray_left[:3], np.array((0.0, 0.0, -1.0))))
    assert(np.isclose(abs(params["view_angles"].angle_left), ray_left_angle))
    assert(ray_left[1] == 0.0)

    ray_down = get_gaze_dir_local(proj, np.array([0.5, 1.0]))
    ray_down_angle = np.arccos(np.dot(ray_down[:3], np.array((0.0, 0.0, -1.0))))
    assert(np.isclose(abs(params["view_angles"].angle_down), ray_down_angle))
    assert(ray_down[0] == 0.0)

    ray_up = get_gaze_dir_local(proj, np.array([0.5, 0.0]))
    ray_up_angle = np.arccos(np.dot(ray_up[:3], np.array((0.0, 0.0, -1.0))))
    assert(np.isclose(abs(params["view_angles"].angle_up), ray_up_angle))
    assert(ray_up[0] == 0.0)

    print("Passed all tests.")    
