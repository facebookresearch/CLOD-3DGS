// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef EYE_TRACKER_EYE_TRACKER_H
#define EYE_TRACKER_EYE_TRACKER_H

#include <vector>

#include <glm/gtc/type_ptr.hpp>

namespace eye_tracker {
  
/**
  * @brief Eye tracking abstract class
  */
class EyeTracker {
 public:
  EyeTracker() = default;
  virtual glm::vec2 getLastEyePosition(uint32_t view_index,
                                       glm::mat4& view_matrix,
                                       glm::mat4& projection_matrix) = 0;
  virtual glm::vec3 getEyeDirection(uint32_t view_index) = 0;

 protected:
  std::vector<glm::vec2> eye_positions_;
};

}

#endif  // EYE_TRACKER_EYE_TRACKER_H