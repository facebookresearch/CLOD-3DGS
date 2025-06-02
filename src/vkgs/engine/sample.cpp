// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "sample.h"


namespace vkgs {

SampleState Interpolate(SampleState& start, SampleState& end,
                               float alpha) {
  uint32_t num_frames = start.pos.size();

  SampleState sample_state{};
  for (uint32_t i = 0; i < num_frames; i++) {
    // interpolate pos and quat
    glm::mat4 start_mat_translation(1);
    start_mat_translation = glm::translate(start_mat_translation, start.pos[i]);
    auto start_mat_rotation = glm::toMat4(start.quat[0]);
    auto start_mat = start_mat_translation * start_mat_rotation;

    glm::mat4 end_mat_translation(1);
    end_mat_translation = glm::translate(end_mat_translation, start.pos[i]);
    auto end_mat_rotation = glm::toMat4(start.quat[0]);
    auto end_mat = end_mat_translation * end_mat_rotation;

    auto mat = glm::interpolate(start_mat, end_mat, alpha);

    auto translation = glm::vec3(mat[3]);
    sample_state.pos.push_back(translation);

    auto rot_mat = glm::mat3(mat);
    sample_state.quat.push_back(glm::quat(rot_mat));

    // interpolate center
    sample_state.center.push_back(
        glm::mix(start.center[i], end.center[i], alpha));

    // interpolate center
    sample_state.gaze_dir.push_back(
        glm::normalize(glm::mix(start.gaze_dir[i], end.gaze_dir[i], alpha)));

    // interpolate view_angles
    core::ViewFrustumAngles view_angle;
    view_angle.angle_right = glm::mix(start.view_angles[i].angle_right,
      end.view_angles[i].angle_right, alpha);
    view_angle.angle_left = glm::mix(
      start.view_angles[i].angle_left, end.view_angles[i].angle_left, alpha);
    view_angle.angle_down = glm::mix(
      start.view_angles[i].angle_down, end.view_angles[i].angle_down, alpha);
    view_angle.angle_up = glm::mix(
      start.view_angles[i].angle_up, end.view_angles[i].angle_up, alpha);
    sample_state.view_angles.push_back(view_angle);
  }

  return sample_state;
};

}  // namespace vkgs
