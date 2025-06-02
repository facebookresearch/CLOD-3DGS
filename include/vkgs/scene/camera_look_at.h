// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_SCENE_CAMERA_LOOK_AT_H
#define VKGS_SCENE_CAMERA_LOOK_AT_H

#include <glm/glm.hpp>

#include "vkgs/scene/camera.h"

namespace vkgs {

class CameraLookAt : public Camera {
 public:
  CameraLookAt();
  ~CameraLookAt();

  /**
   * Set fov and dolly zoom
   *
   * fov: fov Y, in radians
   */
  void SetFov(float fov);

  glm::mat4 ViewMatrix() const;
  virtual glm::vec3 Eye() const;
  
  void Rotate(float x, float y);
  void Translate(float x, float y, float z = 0.f);
  void Zoom(float x);
  void DollyZoom(float scroll);

  void reset();

  float fov() const noexcept { return fovy_; }
  static constexpr float min_fov() { return glm::radians(40.f); }
  static constexpr float max_fov() { return glm::radians(100.f); }

 private:
  float fovy_ = glm::radians(60.f);

  // camera = center + r (sin phi sin theta, cos phi, sin phi cos theta)
  glm::vec3 center_ = {0.f, 0.f, 0.f};
  float r_ = 2.f;
  float phi_ = glm::radians(45.f);
  float theta_ = glm::radians(45.f);

  float rotation_sensitivity_ = 0.01f;
  float translation_sensitivity_ = 0.002f;
  float zoom_sensitivity_ = 0.01f;
  float dolly_zoom_sensitivity_ = glm::radians(1.f);
};

}  // namespace vkgs

#endif  // VKGS_SCENE_CAMERA_H
