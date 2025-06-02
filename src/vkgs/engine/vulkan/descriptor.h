// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_DESCRIPTOR_H
#define VKGS_ENGINE_VULKAN_DESCRIPTOR_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/descriptor_layout.h"

namespace vkgs {
namespace vk {


/**
 * @brief Vulkan descriptor
 */
class Descriptor {
 public:
  Descriptor();

  Descriptor(Context context, DescriptorLayout layout);

  void initialize(Context context, DescriptorLayout layout);
  
  ~Descriptor();

  operator VkDescriptorSet() const;

  void Update(
    uint32_t binding,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize size = 0
 );

  void Update(uint32_t binding, VkSampler sampler,
              std::vector<VkImageView>& image_views, VkImageLayout image_layout,
              VkDeviceSize offset, VkDeviceSize size
 );

  void UpdateInputAttachment(uint32_t binding, VkImageView image_view);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_DESCRIPTOR_H
