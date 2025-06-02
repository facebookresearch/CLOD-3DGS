// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_IMAGE_SPEC_H
#define VKGS_ENGINE_VULKAN_IMAGE_SPEC_H

#include <vulkan/vulkan.h>

namespace vkgs {
namespace vk {


/**
 * @brief Image specification
 */
struct ImageSpec {
  uint32_t width = 0;
  uint32_t height = 0;
  VkImageUsageFlags usage = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_IMAGE_SPEC_H
