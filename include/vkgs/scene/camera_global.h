// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_SCENE_CAMERA_GLOBAL_H
#define VKGS_SCENE_CAMERA_GLOBAL_H

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <core/structs.h>
#include "vkgs/scene/camera.h"

namespace vkgs {

class CameraGlobal : public Camera {
 public:
  virtual glm::mat4 ViewMatrix() const;
  virtual glm::vec3 Eye() const;

  glm::vec3& pos(glm::vec3& p);
  glm::quat& quat(glm::quat& q);
  core::ViewFrustumAngles& view_angles();
  core::ViewFrustumAngles& view_angles(core::ViewFrustumAngles& va);
  core::ViewFrustumAngles& view_angles(float angle_right, float angle_left, float angle_down, float angle_up);

private:
  glm::vec3 pos_;
  glm::quat quat_;

  core::ViewFrustumAngles view_angles_;
};

}  // namespace vkgs

#endif  // VKGS_SCENE_CAMERA_GLOBAL_H
