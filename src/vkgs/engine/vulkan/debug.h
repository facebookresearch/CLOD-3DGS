// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef VKGS_ENGINE_VULKAN_DEBUG_H
#define VKGS_ENGINE_VULKAN_DEBUG_H

#include <vulkan/vulkan.h>
#include "vkgs/engine/vulkan/context.h"

namespace vkgs {
namespace vk {
namespace debug {

/**
 * @brief Set name for Vulkan object
 * This can help with debugging in RenderDoc.
 * @param context Vulkan context
 * @param object_type Vulkan object type
 * @param object_id Vulkan ID
 * @param object_name label for object
 */
void set_object_name(Context& context, VkObjectType object_type,
                     uint64_t object_id, std::string object_name);

/**
 * @brief Insert label into command buffer
 * This is useful for labeling stages in the render process.
 * @param context Vulkan context
 * @param cb Vulkan command buffer
 * @param label label for command buffer
 */
void set_cmd_insert_label(Context& context, VkCommandBuffer& cb,
                          std::string label);

};  // namespace debug
};  // namespace vk
};  // namespace vkgs

#endif  // VKGS_ENGINE_VULKAN_DEBUG_H