// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_ATTACHMENT_H
#define VKGS_ENGINE_VULKAN_ATTACHMENT_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"
#include "vkgs/engine/vulkan/image_spec.h"

namespace vkgs {
namespace vk {

class Context;

/**
 * @brief Vulkan attachment
 */
class Attachment {
 public:
  Attachment();

  Attachment(Context context, uint32_t width, uint32_t height, VkFormat format,
             VkSampleCountFlagBits samples, VkImageUsageFlags flags, uint32_t layers=1);

  ~Attachment();

  operator VkImageView() const;

  VkImage image() const;
  VkImageView image_view() const;
  VkImageUsageFlags usage() const;
  VkFormat format() const;
  ImageSpec image_spec() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_ATTACHMENT_H
