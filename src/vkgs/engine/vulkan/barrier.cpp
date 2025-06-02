// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/barrier.h"

namespace vkgs {
namespace vk {
namespace barrier {

  void changeImageLayout(VkCommandBuffer cb, Context context, VkImage image,
                       VkImageLayout src_layout, VkImageLayout dst_layout,
                       VkAccessFlags src_access, VkAccessFlags dst_access,
                       VkPipelineStageFlags src_pipeline_stage,
                       VkPipelineStageFlags dst_pipeline_stage) {
  if (src_layout == dst_layout) {
    return;
  }
  
  VkImageSubresourceRange range = {};
  range.aspectMask =
      ((src_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL) ||
       (dst_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL))
          ? VK_IMAGE_ASPECT_DEPTH_BIT
          : VK_IMAGE_ASPECT_COLOR_BIT;
  range.baseMipLevel = 0;
  range.levelCount = 1;
  range.baseArrayLayer = 0;
  range.layerCount = 1;

  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.srcAccessMask = src_access;
  barrier.srcAccessMask = dst_access;
  barrier.oldLayout = src_layout;
  barrier.newLayout = dst_layout;
  barrier.srcQueueFamilyIndex = context.graphics_queue_family_index();
  barrier.dstQueueFamilyIndex = context.graphics_queue_family_index();
  barrier.image = image;
  barrier.subresourceRange = range;

  vkCmdPipelineBarrier(cb, src_pipeline_stage, dst_pipeline_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);
}

};  // namespace barrier
};  // namespace vk
};  // namespace vkgs
