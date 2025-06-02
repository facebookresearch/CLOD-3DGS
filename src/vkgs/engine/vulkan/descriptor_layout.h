// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_DESCRIPTOR_LAYOUT_H
#define VKGS_ENGINE_VULKAN_DESCRIPTOR_LAYOUT_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"

namespace vkgs {
namespace vk {


/**
 * @brief DescriptorLayout binding information
 */
struct DescriptorLayoutBinding {
  uint32_t binding = 0;
  VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_SAMPLER;
  uint32_t descriptor_count = 1;
  VkShaderStageFlags stage_flags = 0;
};

/**
 * @brief Collection of DescriptorLayoutBinding
 */
struct DescriptorLayoutCreateInfo {
  std::vector<DescriptorLayoutBinding> bindings;
};


/**
 * @brief Vulkan descriptor layout
 */
class DescriptorLayout {
 public:
  DescriptorLayout();
  DescriptorLayout(Context context,
                   const DescriptorLayoutCreateInfo& create_info);
  ~DescriptorLayout();

  operator VkDescriptorSetLayout() const;

  VkDescriptorType type(uint32_t binding) const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_DESCRIPTOR_LAYOUT_H
