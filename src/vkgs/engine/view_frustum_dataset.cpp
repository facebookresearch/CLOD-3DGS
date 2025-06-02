// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/view_frustum_dataset.h"

namespace vkgs {

/**
 * @brief View frustum dataset constructor
 * @param filename filename of view frustum dataset (.yaml file)
 */
ViewFrustumDataset::ViewFrustumDataset(const std::string& filename) {
  auto views = YAML::LoadFile(filename);
  for (uint32_t i = 0; i < views.size(); i++) {
    auto num_frames = views[i]["sample_state"]["pos"].size();

    ViewFrustum view_frustum{};
    for (uint32_t f = 0; f < num_frames; f++) {
      auto pos = views[i]["sample_state"]["pos"][f].as<std::vector<float>>();
      auto quat = views[i]["sample_state"]["quat"][f].as<std::vector<float>>();
      auto view_angles = views[i]["sample_state"]["view_angles"][f].as<std::vector<float>>();
      auto center = views[i]["sample_state"]["center"][f].as<std::vector<float>>();
      auto gaze_dir = views[i]["sample_state"]["gaze_dir"][f].as<std::vector<float>>();
      
      view_frustum.sample_state.pos.push_back(glm::vec3(pos[0], pos[1], pos[2]));
      view_frustum.sample_state.quat.push_back(
          glm::quat(quat[0], quat[1], quat[2], quat[3]));
      view_frustum.sample_state.view_angles.push_back(core::ViewFrustumAngles{
          view_angles[0], view_angles[1], view_angles[2], view_angles[3]});
      view_frustum.sample_state.center.push_back(glm::vec2(center[0], center[1]));
      view_frustum.sample_state.gaze_dir.push_back(
          glm::vec3(gaze_dir[0], gaze_dir[1], gaze_dir[2]));

      if (views[i]["metadata"]["mode"]) {
        view_frustum.mode = views[i]["metadata"]["mode"].as<std::string>();
      }
    }
    view_frustums_.push_back(view_frustum);
  }
}


/**
 * @brief Get size of dataset
 * @return number of frustums in dataset
 */
size_t ViewFrustumDataset::size() {
  return view_frustums_.size();
}


/**
 * @brief Get view frustum from dataset
 * @param index index of view frustum
 * @return view frustum
 */
const ViewFrustum& ViewFrustumDataset::operator[](uint32_t index) const {
  return view_frustums_[index];
}


glm::mat4 ViewFrustumDataset::TransformationMatrix(uint32_t index) {
  auto view_frustum = view_frustums_[index];
  glm::mat4 mat_translation(1);
  mat_translation = glm::translate(mat_translation, view_frustum.sample_state.pos[0]);
  auto mat_rotation = glm::toMat4(view_frustum.sample_state.quat[0]);
  return mat_translation * mat_rotation;
}


/**
 * @brief Get interpolated transformation matrix
 * @param alpha interpolation factor between 0.0 and 1.0
 * @param num_cameras number of cameras
 * @return interpolated transformation matrix
 */
glm::mat4 ViewFrustumDataset::GetMatrixInterpolated(float alpha, uint32_t num_cameras) {
  alpha = glm::clamp(alpha, 0.0f, 1.0f);
  
  std::vector<uint32_t> cameras;  
  if (num_cameras > 0) { 
    for (uint32_t i = 0; i < num_cameras; i++) {
      uint32_t index = (uint32_t)(i * ((float)view_frustums_.size() / (float)num_cameras));
      cameras.push_back(index);
    }
  } else {
    for (uint32_t i = 0; i < view_frustums_.size(); i++) {
      cameras.push_back(i);
    }
  }
  cameras.push_back(0);

  auto low_index = glm::floor(alpha * (cameras.size()-1));
  auto high_index = glm::ceil(alpha * (cameras.size()-1));

  float delta = 0;
  if (high_index > low_index) {
    delta = ((alpha * (cameras.size()-1)) - low_index) / (high_index - low_index);
  }

  if (low_index == high_index) {
    return TransformationMatrix(cameras[(uint32_t)low_index]);
  }
  auto low_mat = TransformationMatrix(cameras[(uint32_t)low_index]);
  auto high_mat = TransformationMatrix(cameras[(uint32_t)high_index]);
  auto mat = glm::interpolate(low_mat, high_mat, delta);
  return mat;
}

SampleState ViewFrustumDataset::GetSampleStateInterpolated(float alpha, uint32_t num_cameras) {
  SampleState sample_state{};
  auto mat = GetMatrixInterpolated(alpha, num_cameras);
  auto translation = glm::vec3(mat[3]);
  sample_state.pos.push_back(translation);

  auto rot_mat = glm::mat3(mat);
  sample_state.quat.push_back(glm::quat(rot_mat));

  return sample_state;
}

};  // namespace vkgs
