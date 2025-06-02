// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>

#include "foveation/math.h"
#include "vkgs/scene/camera_global.h"

namespace vkgs {

glm::mat4 CameraGlobal::ViewMatrix() const {
  glm::mat4 mat_translation(1);
  mat_translation = glm::translate(mat_translation, pos_);
  auto mat_rotation = glm::toMat4(quat_);
  return glm::inverse(mat_translation * mat_rotation);
}

glm::vec3 CameraGlobal::Eye() const {
  return pos_;
}

glm::vec3& CameraGlobal::pos(glm::vec3& p) {
  pos_ = p;
  return pos_;
}

glm::quat& CameraGlobal::quat(glm::quat& q) {
  quat_ = q;
  return quat_;
}

core::ViewFrustumAngles& CameraGlobal::view_angles() {
  return view_angles_;
}

core::ViewFrustumAngles& CameraGlobal::view_angles(core::ViewFrustumAngles& va) {
  view_angles_ = va;
  return view_angles_;
}

core::ViewFrustumAngles& CameraGlobal::view_angles(
    float angle_right, float angle_left, float angle_down, float angle_up) {
  view_angles_.angle_right = angle_right;
  view_angles_.angle_left = angle_left;
  view_angles_.angle_up = angle_up;
  view_angles_.angle_down = angle_down;
  return view_angles_;
}

}  // namespace vkgs
