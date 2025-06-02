// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <vkgs/scene/camera_look_at.h>

#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>

#include "foveation/math.h"

namespace vkgs {

CameraLookAt::CameraLookAt() {}

CameraLookAt::~CameraLookAt() {}

void CameraLookAt::SetFov(float fov) {
  // dolly zoom
  //r_ *= std::tan(fovy_ / 2.f) / std::tan(fov / 2.f);

  fovy_ = fov;
}

glm::mat4 CameraLookAt::ViewMatrix() const {
  return glm::lookAt(Eye(), center_, glm::vec3(0.f, 1.f, 0.f));
}

glm::vec3 CameraLookAt::Eye() const {
  const auto sin_phi = std::sin(phi_);
  const auto cos_phi = std::cos(phi_);
  const auto sin_theta = std::sin(theta_);
  const auto cos_theta = std::cos(theta_);
  return center_ +
         r_ * glm::vec3(sin_phi * sin_theta, cos_phi, sin_phi * cos_theta);
}

void CameraLookAt::Rotate(float x, float y) {
  theta_ -= rotation_sensitivity_ * x;
  float eps = glm::radians<float>(0.1f);
  phi_ =
      std::clamp(phi_ - rotation_sensitivity_ * y, eps, glm::pi<float>() - eps);
}

void CameraLookAt::Translate(float x, float y, float z) {
  // camera = center + r (sin phi sin theta, cos phi, sin phi cos theta)
  const auto sin_phi = std::sin(phi_);
  const auto cos_phi = std::cos(phi_);
  const auto sin_theta = std::sin(theta_);
  const auto cos_theta = std::cos(theta_);
  center_ +=
      translation_sensitivity_ * r_ *
      (-x * glm::vec3(cos_theta, 0.f, -sin_theta) +
       y * glm::vec3(-cos_phi * sin_theta, sin_phi, -cos_phi * cos_theta) +
       -z * glm::vec3(sin_phi * sin_theta, cos_phi, sin_phi * cos_theta));
}

void CameraLookAt::Zoom(float x) { r_ /= std::exp(zoom_sensitivity_ * x); }

void CameraLookAt::DollyZoom(float scroll) {
  float new_fov = std::clamp(fovy_ - scroll * dolly_zoom_sensitivity_,
                             min_fov(), max_fov());
  SetFov(new_fov);
}

void CameraLookAt::reset() {
  r_ = 2.f;
  phi_ = glm::radians(45.f);
  theta_ = glm::radians(45.f);
}

}  // namespace vkgs
