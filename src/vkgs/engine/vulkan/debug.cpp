// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vkgs/engine/vulkan/debug.h"

namespace vkgs {
namespace vk {
namespace debug {

void set_object_name(Context& context, VkObjectType object_type,
                     uint64_t object_id, std::string object_name){
  VkDebugUtilsObjectNameInfoEXT object_info;
  object_info.sType = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
  object_info.pNext = nullptr;
  object_info.objectType = object_type;
  object_info.objectHandle = object_id;
  object_info.pObjectName = object_name.c_str();
  auto func =
      (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(
      context.instance(), "vkSetDebugUtilsObjectNameEXT");
  func(context.device(), &object_info);
}

void set_cmd_insert_label(Context& context, VkCommandBuffer& cb, std::string label) {
  VkDebugUtilsLabelEXT label_info = {};
  label_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
  label_info.pNext = nullptr;
  label_info.pLabelName = label.c_str();

  auto func = (PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      context.instance(), "vkCmdInsertDebugUtilsLabelEXT");
  func(cb, &label_info);
}

};  // namespace debug
};  // namespace vk
};  // namespace vkgs
