// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_SCENE_CAMERA_H
#define VKGS_SCENE_CAMERA_H

#include <glm/glm.hpp>

namespace vkgs {

/**
 * @brief Camera data structure
 */
class Camera {
 public:
  Camera();
  ~Camera();

  auto Near() const noexcept { return near_; }
  auto Far() const noexcept { return far_; }

  void SetWindowSize(uint32_t width, uint32_t height);

  glm::mat4 ProjectionMatrix(float fov, float width, float height, float res_x, float res_y, float cx=0.0, float cy=0.0) const;
  glm::mat4 ProjectionMatrixXR(float angle_right, float angle_left,
                               float angle_down, float angle_up, float width,
                               float height, float res_x, float res_y,
                               float cx = 0.0, float cy = 0.0) const;
  virtual glm::mat4 ViewMatrix() const = 0;
  virtual glm::vec3 Eye() const = 0;

  uint32_t width() const noexcept { return width_; }
  uint32_t height() const noexcept { return height_; }

 protected:
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  float near_ = 0.01f;
  float far_ = 100.f;
};

}  // namespace vkgs

#endif  // VKGS_SCENE_CAMERA_H
