// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_SAMPLE_H
#define VKGS_ENGINE_SAMPLE_H

#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_interpolation.hpp>

#include "core/structs.h"


namespace vkgs {

/**
 * @brief Rendering parameters for rendering frames from Python
 */
struct SampleParams {
  uint32_t num_frames_benchmark = 0;
  uint32_t num_frames_recorder = 0;
  std::vector<std::vector<float>> lod;             // per frame, per level
  std::vector<std::vector<float>> res;             // per frame, per level
  std::vector<std::vector<glm::vec4>> lod_params;  // per frame
};

/**
 * @brief Camera state for rendering frames from Python
 */
struct SampleState {
  std::vector<glm::vec3> pos;                        // per frame
  std::vector<glm::quat> quat;                       // per frame
  std::vector<glm::vec2> center;                     // per frame
  std::vector<glm::vec3> gaze_dir;                   // per frame
  std::vector<core::ViewFrustumAngles> view_angles;  // per frame
};

/**
 * @brief Rendering results for rendering frames from Python
 */
struct SampleResult {
  std::vector<float> time;                          // per frame, in milliseconds
  std::vector<uint32_t> shape;                      // image shape
  std::vector<unsigned char> data;                  // image data
};

}  // namespace vkgs

#endif  // VKGS_ENGINE_SAMPLE_H