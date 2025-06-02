// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/mesh/frustum_mesh.h"

namespace vkgs {
namespace vk {

FrustumMesh::FrustumMesh(Context& context, ViewFrustum& view_frustum)
    : Mesh(context) {
  float distance = 0.2;
  float distance_gaze = 0.3;

  std::vector<glm::vec3> points = {
      glm::vec3(0.0, 0.0, 0.0), glm::vec3(1.0, 1.0, 1.0),
      glm::vec3(1.0, -1.0, 1.0), glm::vec3(-1.0, -1.0, 1.0),
      glm::vec3(-1.0, 1.0, 1.0)};
  float width = glm::tan(view_frustum.sample_state.view_angles[0].angle_right) -
                glm::tan(view_frustum.sample_state.view_angles[0].angle_left);
  float height = glm::tan(view_frustum.sample_state.view_angles[0].angle_up) -
                 glm::tan(view_frustum.sample_state.view_angles[0].angle_down);
  
  glm::mat4 mat_translation(1);
  mat_translation = glm::translate(mat_translation, view_frustum.sample_state.pos[0]);
  auto mat_rotation = glm::toMat4(view_frustum.sample_state.quat[0]);
  auto cam_mat = mat_translation * mat_rotation;
  auto proj_mat = fov::math::perspective_asymmetric(
    view_frustum.sample_state.view_angles[0].angle_right,
    view_frustum.sample_state.view_angles[0].angle_left,
    view_frustum.sample_state.view_angles[0].angle_down,
    view_frustum.sample_state.view_angles[0].angle_up,
    width,
    height,
    width,
    height,
    0.01,
    100.0,
    0.5f,
    0.5f
  );
  
  // inverse projection matrix
  for (uint32_t i = 1; i < points.size(); i++) {
    points[i] = glm::vec3(glm::inverse(proj_mat) * glm::vec4(points[i], 1));
    points[i] = points[i] / glm::distance(glm::vec3(0), points[i]);
    points[i] = points[i] * distance;
  }

  // camera transformation matrix
  for (uint32_t i = 0; i < points.size(); i++) {
    points[i] = glm::vec3(cam_mat * glm::vec4(points[i], 1.0));
  }

  // get gaze direction vector
  auto gaze_dir = view_frustum.sample_state.gaze_dir[0] * distance_gaze;
  auto gaze_point = points[0] + gaze_dir;

  position_ = {
      points[0].x,  points[0].y,  points[0].z,
      points[1].x,  points[1].y,  points[1].z,
      points[2].x,  points[2].y,  points[2].z,
      points[3].x,  points[3].y,  points[3].z,
      points[4].x,  points[4].y,  points[4].z,
      points[0].x,  points[0].y,  points[0].z,
      gaze_point.x, gaze_point.y, gaze_point.z,
  };

  auto f_color = glm::vec3(1.0f, 1.0f, 1.0f);
  if (view_frustum.mode == std::string("active")) {
    f_color = glm::vec3(0.0f, 1.0f, 0.0f);
  }
  color_ = {
      f_color.x,  f_color.y,  f_color.z,  1.f,
      f_color.x,  f_color.y,  f_color.z,  1.f,
      f_color.x,  f_color.y,  f_color.z,  1.f,
      f_color.x,  f_color.y,  f_color.z,  1.f,
      f_color.x,  f_color.y,  f_color.z,  1.f,
      1.f,        0.f,        0.f,        1.f,
      1.f,        0.f,        0.f,        1.f,
  };

  index_ = {
      0, 1,
      0, 2,
      0, 3,
      0, 4,
      1, 2,
      2, 3,
      3, 4,
      4, 1,
      5, 6,
  };
  position_buffer_ = vk::Buffer(
      context, position_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      "buffer_axis_position_buffer");
  color_buffer_ = vk::Buffer(
      context, color_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      "buffer_axis_color_buffer");
  index_buffer_ = vk::Buffer(
      context, index_.size() * sizeof(float),
      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      "buffer_axis_index_buffer");
}

}  // namespace vk
}  // namespace vkgs
