// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_RENDER_PASS_H
#define VKGS_ENGINE_VULKAN_RENDER_PASS_H

#include <memory>

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"

namespace vkgs {
namespace vk {

enum class RenderPassType {
  NORMAL,
  OIT,
};

/**
 * @brief Vulkan render pass
 */
class RenderPass {
 public:
  RenderPass();

  RenderPass(
      Context context, VkSampleCountFlagBits samples, VkFormat color_format,
      VkFormat depth_format, uint32_t num_views,
      VkImageLayout color_final_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

  ~RenderPass();

  operator VkRenderPass() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_RENDER_PASS_H
