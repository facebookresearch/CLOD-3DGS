// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_BARRIER_H
#define VKGS_ENGINE_VULKAN_BARRIER_H

#include <vulkan/vulkan.h>

#include "vkgs/engine/vulkan/context.h"

namespace vkgs {
namespace vk {
namespace barrier {

// Adapted from
// https://gitlab.kitware.com/iMSTK/iMSTK/-/blob/v2.0.0/Source/Rendering/VulkanRenderer/imstkVulkanUtilities.h

void changeImageLayout(VkCommandBuffer cb, Context context, VkImage image,
                       VkImageLayout src_layout, VkImageLayout dst_layout,
                       VkAccessFlags src_access, VkAccessFlags dst_access,
                       VkPipelineStageFlags src_pipeline_stage,
                       VkPipelineStageFlags dst_pipeline_stage);

};  // namespace barrier
};  // namespace vk
};  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_BARRIER_H