// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef EYE_TRACKER_EYE_TRACKER_OPENXR_H
#define EYE_TRACKER_EYE_TRACKER_OPENXR_H

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>

#include "eye_tracker/eye_tracker.h"

namespace eye_tracker {
  
/**
 * @brief Eye tracker using OpenXR
 */
class EyeTrackerOpenXR : public EyeTracker {
 public:
  EyeTrackerOpenXR(XrInstance& xr_instance, XrSession& xr_session);
  void update(XrFrameState xr_frame_state);

  virtual glm::vec2 getLastEyePosition(uint32_t view_index,
                                       glm::mat4& view_matrix,
                                       glm::mat4& projection_matrix);

  virtual glm::vec3 getEyeDirection(uint32_t view_index);

  const bool status();

private:
  glm::vec2 projectEyeGaze(XrPosef& eye_pose, glm::mat4& view_matrix,
                           glm::mat4& projection_matrix);

  XrEyeTrackerFB xr_eye_tracker_{};
  XrEyeGazesFB xr_eye_gazes_{XR_TYPE_EYE_GAZES_FB};
  std::vector<XrPosef> xr_eye_poses_last_;
  XrSpace xr_screen_space_;
  XrResult status_;
  
  PFN_xrCreateEyeTrackerFB xrCreateEyeTrackerFB;
  PFN_xrGetEyeGazesFB xrGetEyeGazesFB;
};

}

#endif  // EYE_TRACKER_EYE_TRACKER_OPENXR_H