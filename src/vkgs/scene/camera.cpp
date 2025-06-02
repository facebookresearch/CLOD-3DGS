// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <vkgs/scene/camera.h>

#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>

#include "foveation/math.h"

namespace vkgs {

Camera::Camera() {}

Camera::~Camera() {}

void Camera::SetWindowSize(uint32_t width, uint32_t height) {
  width_ = width;
  height_ = height;
}

glm::mat4 Camera::ProjectionMatrix(float fov, float width, float height,
                                   float res_x, float res_y, float cx,
                                   float cy) const {
  auto projection = fov::math::perspective_symmetric(
      fov, width, height, res_x, res_y, near_, far_, cx, cy);
  
  // gl to vulkan projection matrix
  glm::mat4 conversion = glm::mat4(1.f);
  conversion[1][1] = -1.f;
  conversion[2][2] = 0.5f;
  conversion[3][2] = 0.5f;
  return conversion * projection;
}

glm::mat4 Camera::ProjectionMatrixXR(float angle_right, float angle_left,
                             float angle_down, float angle_up, float width,
                             float height, float res_x, float res_y,
                             float cx, float cy) const {
  glm::mat4 projection = fov::math::perspective_asymmetric(
      angle_right, angle_left, angle_down, angle_up, width, height, res_x,
      res_y, near_, far_, cx, cy);

  // gl to vulkan projection matrix
  glm::mat4 conversion = glm::mat4(1.f);
  conversion[1][1] = -1.f;
  conversion[2][2] = 0.5f;
  conversion[3][2] = 0.5f;
  return conversion * projection;
}

}  // namespace vkgs
