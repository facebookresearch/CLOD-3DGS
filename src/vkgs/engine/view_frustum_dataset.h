// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_VIEW_FRUSTUM_DATASET_H
#define VKGS_ENGINE_VULKAN_VIEW_FRUSTUM_DATASET_H

#include <cmath>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_interpolation.hpp>

#include "core/structs.h"
#include "vkgs/engine/sample.h"


namespace vkgs {


/**
 * @brief View frustum data structure
 */
struct ViewFrustum {
  SampleState sample_state; // camera state
  std::string mode;         // camera mode label
};
  

/**
 * @brief View frustum dataset
 */
class ViewFrustumDataset {
 public:
  ViewFrustumDataset() = delete;
  ViewFrustumDataset(const std::string& filename);
  ~ViewFrustumDataset() = default;

  size_t size();
  const ViewFrustum& operator[](uint32_t index) const;
  glm::mat4 TransformationMatrix(uint32_t index);
  glm::mat4 GetMatrixInterpolated(float alpha, uint32_t num_cameras=0);
  SampleState GetSampleStateInterpolated(float alpha, uint32_t num_cameras);

 private:
  std::string filename_;
  std::vector<ViewFrustum> view_frustums_;
};

}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_VIEW_FRUSTUM_DATASET_H