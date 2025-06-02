// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "foveation/math.h"

#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace fov {
namespace math {
  
/**
 * @brief Perspective symmetric projection matrix constructor
 * @param fovy field of view for Y-axis (height)
 * @param width FOV width
 * @param height FOV height
 * @param res_x image width
 * @param res_y image height
 * @param near near plane
 * @param far far plane
 * @param cx center x
 * @param cy center y
 * @return projection matrix
 */
glm::mat4 perspective_symmetric(
    float fovy,   // field of view for Y-axis (height)
    float width,  // fov width
    float height, // fov height
    float res_x,  // image width
    float res_y,  // image height
    float near,   // near plane
    float far,    // far plane
    float cx,     // center x
    float cy      // center y
  ) {
  // math for off-center perspective projection
  // https://computergraphics.stackexchange.com/a/5058
  float aspect = static_cast<float>(width) / height;
  float f = 1.0f / tanf(fovy / 2.0f);
  
  // pixel space [0, W], [0, H]
  float top = (cy * res_y) + (height * 0.5);
  float bottom = (cy * res_y) - (height * 0.5);
  float right = (cx * res_x) + (width * 0.5);
  float left = (cx * res_x) - (width * 0.5);
  
  // normalized space [0, 1], [0, 1]
  top = ((top / res_y) * 2.0) - 1.0;
  bottom = ((bottom / res_y) * 2.0) - 1.0;
  right = ((right / res_x) * 2.0) - 1.0;
  left = ((left / res_x) * 2.0) - 1.0;

  float A = (right + left) / (right - left);
  float B = (top + bottom) / (top - bottom);
  
  float data[16] = {
    f / aspect, 0.0f, 0.0f, 0.0f,
    0.0f, f, 0.0f, 0.0f,
    A, -B, (far+near)/(near-far), -1.0f,
    0.0f, 0.0f, 2.0f * far * near / (near - far), 0.0f,
  };
  auto mat = glm::make_mat4(data);
  return mat;
}

/**
 * @brief 
 * @param angle_right right angle of view frustum (radians)
 * @param angle_left left angle of view frustum (radians)
 * @param angle_down down angle of view frustum (radians)
 * @param angle_up up angle of view frustum (radians)
 * @param width FOV width
 * @param height FOV height
 * @param height 
 * @param res_x image width
 * @param res_y image height
 * @param near near plane
 * @param far far plane
 * @param cx center x
 * @param cy center y
 * @return projection matrix
 */
glm::mat4 perspective_asymmetric(float angle_right,  // angle right
                                 float angle_left,   // angle left
                                 float angle_down,   // angle down
                                 float angle_up,     // angle up
                                 float width,        // fov width
                                 float height,       // fov height
                                 float res_x,        // image height
                                 float res_y,        // image height
                                 float near,         // near plane
                                 float far,          // far plane
                                 float cx,           // center x
                                 float cy            // center y
) {
  // adapted from
  // https://amini-allight.org/post/openxr-tutorial-addendum-2-single-swapchain
  float angle_width = tan(angle_right) - tan(angle_left);
  float angle_height = tan(angle_up) - tan(angle_down);

  float aspect = static_cast<float>(width) / height;

  auto top_bound = tan(angle_up);
  auto bottom_bound = tan(angle_down);
  auto right_bound = tan(angle_right);
  auto left_bound = tan(angle_left);

  float cy_inv = 1.0 - cy;
  
  // pixel space [0, W], [0, H]
  float top = (cy_inv * res_y) + (height * 0.5);
  float bottom = (cy_inv * res_y) - (height * 0.5);
  float right = (cx * res_x) + (width * 0.5);
  float left = (cx * res_x) - (width * 0.5);

  // normalized space [0, 1], [0, 1]
  top = top / res_y;
  bottom = bottom / res_y;
  right = right / res_x;
  left = left / res_x;

  top = (top * angle_height) + bottom_bound;
  bottom = (bottom * angle_height) + bottom_bound;
  left = (left * angle_width) + left_bound;
  right = (right * angle_width) + left_bound;

  auto fov_angle_height = top - bottom;
  auto fov_angle_width = right - left;

  glm::mat4 data = glm::mat4(0.0f);
  data[0][0] = 2.0f / fov_angle_width;
  data[2][0] = (right + left) / fov_angle_width;
  data[1][1] = 2.0f / fov_angle_height;
  data[2][1] = (top + bottom) / fov_angle_height;
  data[2][2] = -far / (far - near);
  data[3][2] = -(far * near) / (far - near);
  data[2][3] = -1;

  return data;
}

}
}