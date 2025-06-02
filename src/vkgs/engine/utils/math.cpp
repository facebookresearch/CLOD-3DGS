// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/utils/math.h"

namespace vkgs {
namespace math {

glm::mat4 ToScaleMatrix4(float s) {
  glm::mat4 m(1.f);
  m[0][0] = s;
  m[1][1] = s;
  m[2][2] = s;
  return m;
}

glm::mat4 ToTranslationMatrix4(const glm::vec3& t) {
  glm::mat4 m(1.f);
  m[3][0] = t[0];
  m[3][1] = t[1];
  m[3][2] = t[2];
  return m;
}

glm::mat4 coordToEyeMatrix(
  const glm::vec2& coord,
  const float z_near,
  const float z_far,
  const glm::mat4& view_matrix,
  const glm::mat4& projection_matrix
) {
  throw std::exception("This function is not properly tested.");

  // get directional vector
  auto inv_proj_view = glm::inverse(projection_matrix * view_matrix);

  auto viewport = glm::vec4(0, 0, 1920, 1080);
  auto coord_norm = coord * glm::vec2(1920, 1080);
  auto start = glm::unProjectZO(glm::vec3(coord_norm, 0.0), view_matrix, projection_matrix, viewport);
  auto end = glm::unProjectZO(glm::vec3(coord_norm, 1.0), view_matrix, projection_matrix, viewport);
  auto forward = glm::normalize(end - start);
  
  // create matrix
  auto inv_view = glm::inverse(view_matrix);
  auto eye = glm::vec3(inv_view[3]);
  auto forward_test = glm::normalize(glm::vec3(inv_view * glm::vec4(0, 0, -1, 0)));
  auto up = glm::normalize(glm::vec3(inv_view * glm::vec4(0, 1, 0, 0)));
  auto right = glm::cross(forward, up);
  up = glm::cross(right, forward);

  glm::mat4 output(1.0);
  output[0][0] = right.x;
  output[0][1] = right.y;
  output[0][2] = right.z;
  output[0][3] = 0;

  output[1][0] = up.x;
  output[1][1] = up.y;
  output[1][2] = up.z;
  output[1][3] = 0;

  output[2][0] = forward.x;
  output[2][1] = forward.y;
  output[2][2] = forward.z;
  output[2][3] = 0;

  output[3][0] = eye.x;
  output[3][1] = eye.y;
  output[3][2] = eye.z;
  output[3][3] = 1;

  output = view_matrix * output;
  return output;
}

};  // namespace math
};  // namespace vkgs
