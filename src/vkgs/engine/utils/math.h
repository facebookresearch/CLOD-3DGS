// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_NATH_H
#define VKGS_ENGINE_MATH_H

#include <set>
#include <string>
#include <sstream>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>


namespace vkgs {
namespace math {

/// @brief Helper function to create a matrix from a uniform scale.
/// @param s A uniform scale factor.
/// @return Scale matrix.
glm::mat4 ToScaleMatrix4(float s);

/// @brief Helper function to create a matrix from a translation.
/// @param t translation vector.
/// @return Translation matrix.
glm::mat4 ToTranslationMatrix4(const glm::vec3& t);

/// @brief Convert 2D coordinate to eye matrix.
/// @param c 2D coordinate. The coordinate should be between 0 and 1.
/// @param z_near Near clipping plane.
/// @param z_far Far clipping plane.
/// @param view_matrix View matrix.
/// @param projection_matrix Projection matrix.
/// @return Eye matrix relative to view matrix.
glm::mat4 coordToEyeMatrix(
  const glm::vec2& coord,
  const float z_near,
  const float z_far,
  const glm::mat4& view_matrix,
  const glm::mat4& projection_matrix);

};
};  // namespace

#endif  // VKGS_ENGINE_MATH_H