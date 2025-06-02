// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "eye_tracker/eye_tracker_openxr.h"

namespace eye_tracker {

/**
 * @brief Constructor for the OpenXR eye tracker
 * @param xr_instance XR instance
 * @param xr_session XR session
 */
EyeTrackerOpenXR::EyeTrackerOpenXR(XrInstance& xr_instance, XrSession& xr_session) {
  // get functions
  {
    PFN_xrVoidFunction func;
    xrGetInstanceProcAddr(xr_instance, "xrCreateEyeTrackerFB", &func);
    xrCreateEyeTrackerFB = (PFN_xrCreateEyeTrackerFB)func;
  }

  {
    PFN_xrVoidFunction func;
    xrGetInstanceProcAddr(xr_instance, "xrGetEyeGazesFB", &func);
    xrGetEyeGazesFB = (PFN_xrGetEyeGazesFB)func;
  }

  // create eye tracker
  XrEyeTrackerCreateInfoFB eye_tracker_create_info{};
  eye_tracker_create_info.type = XR_TYPE_EYE_TRACKER_CREATE_INFO_FB;
  status_ = xrCreateEyeTrackerFB(xr_session, &eye_tracker_create_info, &xr_eye_tracker_);
  
  eye_positions_.resize(2, glm::vec2(0.5));
  xr_eye_poses_last_.resize(2, {});

  XrReferenceSpaceCreateInfo space_create_info{};
  space_create_info.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO;
  space_create_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
  space_create_info.poseInReferenceSpace = {{0, 0, 0, 1}, {0, 0, 0}};
  xrCreateReferenceSpace(xr_session, &space_create_info, &xr_screen_space_);
}

/**
 * @brief Updates eye tracking data
 * @param xr_frame_state OpenXR frame state
 */
void EyeTrackerOpenXR::update(
  XrFrameState xr_frame_state
) {
  XrEyeGazesInfoFB eye_gazes_info{};
  eye_gazes_info.type = XR_TYPE_EYE_GAZES_INFO_FB;
  eye_gazes_info.baseSpace = xr_screen_space_;
  eye_gazes_info.time = xr_frame_state.predictedDisplayTime;

  xrGetEyeGazesFB(xr_eye_tracker_, &eye_gazes_info, &xr_eye_gazes_);

  if (xr_eye_gazes_.gaze[XR_EYE_POSITION_LEFT_FB].isValid) {
    xr_eye_poses_last_[0] = xr_eye_gazes_.gaze[XR_EYE_POSITION_LEFT_FB].gazePose;
  }

  if (xr_eye_gazes_.gaze[XR_EYE_POSITION_RIGHT_FB].isValid) {
    xr_eye_poses_last_[1] = xr_eye_gazes_.gaze[XR_EYE_POSITION_LEFT_FB].gazePose;
  }
}


/**
 * @brief Get the 2D location of the eye gaze location
 * @param view_index view (eye) index
 * @param view_matrix view matrix for the eye
 * @param projection_matrix projection matrix for the eye
 * @return 2D position of the eye tracker, normalized between 0 and 1
 */
glm::vec2 EyeTrackerOpenXR::getLastEyePosition(uint32_t view_index,
                                               glm::mat4& view_matrix,
                                               glm::mat4& projection_matrix) {
  eye_positions_[view_index] =
      projectEyeGaze(xr_eye_poses_last_[view_index], view_matrix, projection_matrix);
  return eye_positions_[view_index];
}

glm::vec2 EyeTrackerOpenXR::projectEyeGaze(
  XrPosef& eye_pose,
  glm::mat4& view_matrix,
  glm::mat4& projection_matrix
) {
  auto eye_pos = glm::vec3(eye_pose.position.x, eye_pose.position.y, eye_pose.position.z);

  auto eye_quat = glm::quat(eye_pose.orientation.w, eye_pose.orientation.x,
                            eye_pose.orientation.y, eye_pose.orientation.z);
  glm::vec4 forward(0.0, 0.0, -1.0, 1.0);
  auto end_pos = glm::vec3(glm::mat4_cast(eye_quat) * forward);
  auto eye_point_3d = end_pos - eye_pos;

  auto transformed_point = projection_matrix * glm::vec4(eye_point_3d, 1.0);
  auto point_2d = glm::vec2(transformed_point) / transformed_point.w;
  auto norm_point_2d = (point_2d + glm::vec2(1.0)) * glm::vec2(0.5);
  return norm_point_2d;
}


/**
 * @brief Get eye direction in view space
 * (0, 0, -1) is the eye direction looking at the center of the screen
 * @param view_index The view (eye) index
 * @return eye direction in view space
 */
glm::vec3 EyeTrackerOpenXR::getEyeDirection(
  uint32_t view_index
) {
  auto eye_pose = xr_eye_poses_last_[view_index];

  auto eye_pos =
      glm::vec3(eye_pose.position.x, eye_pose.position.y, eye_pose.position.z);

  auto eye_quat = glm::quat(eye_pose.orientation.w, eye_pose.orientation.x,
                            eye_pose.orientation.y, eye_pose.orientation.z);
  glm::vec4 forward(0.0, 0.0, -1.0, 1.0);
  auto end_pos = glm::vec3(glm::mat4_cast(eye_quat) * forward);
  auto eye_point_3d = end_pos - eye_pos;
  return eye_point_3d;
}


/**
 * @brief Status of eye tracker
 * @return eye tracker status
 */
const bool EyeTrackerOpenXR::status() { return status_; }

}