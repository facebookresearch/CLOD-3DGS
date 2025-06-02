// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_SHADER_UNIFORMS_H
#define VKGS_ENGINE_VULKAN_SHADER_UNIFORMS_H

#include <glm/glm.hpp>

namespace vkgs {
namespace vk {
namespace shader {

/**
 * @brief Uniform buffer struct for camera
 */
struct alignas(128) Camera {
  glm::mat4 projection;
  glm::mat4 view;
  glm::mat4 eye;
  glm::vec3 camera_position;
  alignas(16) glm::uvec2 screen_size;
  float z_near;
  float z_far;
  float frustum_pad_x = 1.3f;
  float frustum_pad_y = 1.3f;
};


/**
 * @brief Uniform buffer splat information
 */
struct alignas(64) SplatInfo {
  uint32_t point_count;
};

}  // namespace shader
}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_SHADER_UNIFORMS_H
